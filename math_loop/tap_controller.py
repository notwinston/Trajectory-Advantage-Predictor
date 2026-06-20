"""TAP v1 data-collection controller: 2 chains x 6 states x 6 candidates.

Drives the spec's compressed collection loop (section "DATA-COLLECTION LOOP"):

* For each chain, walk 6 policy states. At each state save the before-state
  (adapter + optimizer + step/seed/lr/grpo_beta), measure the probes, and record
  the 16-value policy fingerprint.
* Generate 6 candidate GRPO batches (2 prompts x 4 completions = 8 trajectories
  each), branch every candidate from the byte-identical before-state, label it,
  and write the raw-artifact tree :mod:`math_loop.features` consumes.
* After all 6 candidates are labeled, advance the main chain with a
  SEEDED-RANDOM candidate (keeps the collected history unbiased) and keep the
  last 4 applied updates as history.

Parallel branch dispatch uses a ``ProcessPoolExecutor`` with one worker per GPU
pinned through ``CUDA_VISIBLE_DEVICES`` (4xH100 => 1 main + 3 workers); it falls
back to serial when ``--gpu-count`` is small. All GPU/prime-rl execution is Wave
2 — ``--dry-run`` only plans, makes NO network calls, and imports nothing heavy.
Module-level imports are stdlib + pure ``math_loop`` modules so
``import math_loop.tap_controller`` works with torch/transformers/prime absent.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from math_loop import branch
from math_loop.prime_rl_config import PrimeRLConfigSpec, write_prime_rl_config
from math_loop.schedule import TapCandidate, build_tap_schedule

TRAJECTORIES_PER_CANDIDATE = 8  # 2 prompts x 4 completions
HISTORY_WINDOW = 4


# --- plan model --------------------------------------------------------------

@dataclass(frozen=True)
class CollectionPlan:
    chains: int
    states_per_chain: int
    candidates_per_state: int
    prompts_per_candidate: int
    completions_per_prompt: int
    schedule: list[TapCandidate]
    selected_per_state: dict[str, int]
    worker_gpus: list[int]
    already_labeled: set[str]

    @property
    def total_labels(self) -> int:
        return self.chains * self.states_per_chain * self.candidates_per_state

    @property
    def trajectories_per_candidate(self) -> int:
        return self.prompts_per_candidate * self.completions_per_prompt

    @property
    def total_trajectories(self) -> int:
        return self.total_labels * self.trajectories_per_candidate


def select_main_chain_candidate(
    chain_index: int, state_index: int, candidates_per_state: int, seed: int
) -> int:
    """Seeded-random index of the candidate applied to advance the main chain."""
    rng = random.Random(seed + 7_000_000 + chain_index * 100_000 + state_index)
    return rng.randrange(candidates_per_state)


def worker_gpu_ids(gpu_count: int) -> list[int]:
    """Worker GPU ids (GPU 0 is the main chain; 1..N-1 are branch workers)."""
    if gpu_count <= 1:
        return []
    return list(range(1, gpu_count))


def _prompt_ids_for_plan(args: argparse.Namespace) -> list[str]:
    """Prompt ids for schedule construction WITHOUT touching the network.

    Uses a local ``train_pool.jsonl`` when present; otherwise synthesizes
    placeholder ids so ``--dry-run`` never triggers a HuggingFace download.
    """
    train_path = Path(args.data_dir) / "train_pool.jsonl"
    if train_path.exists():
        from math_loop.data import read_jsonl

        ids = [row["id"] for row in read_jsonl(train_path) if "id" in row]
        if len(ids) >= args.prompts_per_candidate:
            return ids
    pool_size = max(args.prompts_per_candidate * 8, 64)
    return [f"placeholder-prompt-{i:04d}" for i in range(pool_size)]


def existing_labeled_keys(raw_root: str | Path, schedule: list[TapCandidate]) -> set[str]:
    """candidate_ids already labeled (a ``probe_after.json`` exists) for resume."""
    done: set[str] = set()
    for candidate in schedule:
        cand_dir = branch.candidate_dir(raw_root, candidate.state_id, candidate.candidate_index)
        if (cand_dir / "probe_after.json").exists():
            done.add(candidate.candidate_id)
    return done


def build_plan(args: argparse.Namespace) -> CollectionPlan:
    prompt_ids = _prompt_ids_for_plan(args)
    schedule = build_tap_schedule(
        prompt_ids,
        chains=args.chains,
        states_per_chain=args.states_per_chain,
        candidates_per_state=args.candidates_per_state,
        prompts_per_candidate=args.prompts_per_candidate,
        seed=args.seed,
    )
    selected: dict[str, int] = {}
    for chain_index in range(args.chains):
        for state_index in range(args.states_per_chain):
            state_id = f"{chain_index}-{state_index}"
            selected[state_id] = select_main_chain_candidate(
                chain_index, state_index, args.candidates_per_state, args.seed
            )
    already = existing_labeled_keys(args.raw_root, schedule) if Path(args.raw_root).exists() else set()
    return CollectionPlan(
        chains=args.chains,
        states_per_chain=args.states_per_chain,
        candidates_per_state=args.candidates_per_state,
        prompts_per_candidate=args.prompts_per_candidate,
        completions_per_prompt=args.completions_per_prompt,
        schedule=schedule,
        selected_per_state=selected,
        worker_gpus=worker_gpu_ids(args.gpu_count),
        already_labeled=already,
    )


def render_plan(plan: CollectionPlan, args: argparse.Namespace) -> str:
    lines: list[str] = []
    lines.append("TAP v1 data-collection plan")
    lines.append("=" * 48)
    lines.append(
        f"{plan.chains} chains x {plan.states_per_chain} states x "
        f"{plan.candidates_per_state} candidates = {plan.total_labels} labeled candidate updates"
    )
    lines.append(
        f"{plan.trajectories_per_candidate} trajectories/candidate "
        f"({plan.prompts_per_candidate} prompts x {plan.completions_per_prompt} completions) "
        f"=> {plan.total_trajectories} trajectories total"
    )
    lines.append(f"history window: last {HISTORY_WINDOW} applied updates")
    if plan.worker_gpus:
        lines.append(
            f"dispatch: GPU 0 = main chain; branch workers = GPUs {plan.worker_gpus} "
            f"(ProcessPoolExecutor, CUDA_VISIBLE_DEVICES-pinned)"
        )
    else:
        lines.append("dispatch: serial (gpu-count<=1; no worker GPUs)")
    lines.append(f"already labeled (resume-safe skip): {len(plan.already_labeled)}/{plan.total_labels}")
    lines.append("")

    for chain_index in range(plan.chains):
        lines.append(f"chain {chain_index}:")
        for state_index in range(plan.states_per_chain):
            state_id = f"{chain_index}-{state_index}"
            chosen = plan.selected_per_state[state_id]
            worker = plan.worker_gpus[state_index % len(plan.worker_gpus)] if plan.worker_gpus else 0
            lines.append(
                f"  state {state_id}: branch candidates 0..{plan.candidates_per_state - 1} "
                f"-> label all -> apply seeded-random candidate cand_{chosen} (worker GPU {worker})"
            )
    lines.append("")

    sample = plan.schedule[0]
    sample_config = Path(args.config_dir) / "branches" / f"{sample.state_id}_cand_{sample.candidate_index}.toml"
    command = branch.branch_command(args.rl_command, sample_config, resume_step=0)
    lines.append("sample branch command (one GRPO step via prime-rl resume):")
    lines.append("  " + shlex.join(command))
    lines.append(
        f"sample candidate {sample.candidate_id}: prompts={list(sample.prompt_ids)}"
    )
    lines.append("")
    lines.append("NO GPU / NO Prime Intellect calls are made in this wave (Wave 1 = CPU validation).")
    return "\n".join(lines)


# --- rollout extraction (prime-rl train_rollouts.jsonl -> TAP trajectory rows) -

def _rollout_jsonls(branch_output: Path) -> list[Path]:
    """Locate prime-rl's persisted rollout jsonl(s) for a branch run.

    prime-rl writes ``<run>/rollouts/step_<N>/train_rollouts.jsonl`` and the
    run dir may be ``branch_output`` or ``branch_output/run_default`` (Mark's
    fresh-branch-weights layout). We search both, preferring train_rollouts.
    """
    roots = [branch_output / "run_default" / "rollouts", branch_output / "rollouts"]
    found: list[Path] = []
    for root in roots:
        if root.exists():
            found.extend(sorted(root.rglob("train_rollouts.jsonl")))
    if not found:
        for root in roots:
            if root.exists():
                found.extend(sorted(root.rglob("*.jsonl")))
    return found


def _scalar_reward(value: Any) -> float:
    if isinstance(value, dict):
        value = value.get("reward", value.get("score", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _map_rollout_row(row: dict[str, Any], index: int, candidate_id: str) -> dict[str, Any]:
    """Map one prime-rl rollout dict to a features._trajectory_row input.

    Per-token logprobs/entropy are NOT persisted by prime-rl (hardcoded 0.0 in
    trajectories.py), so we leave those absent and let features.py fall back to
    carried scalars / 0.0. Reward + advantage + completion ARE available.
    """
    metrics = row.get("metrics") or row.get("reward_metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}
    completion = row.get("completion", row.get("completion_text", ""))
    if isinstance(completion, list):  # chat messages -> concatenated text
        completion = " ".join(
            str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in completion
        )
    prompt = row.get("prompt", row.get("prompt_text", row.get("question", "")))
    if isinstance(prompt, list):
        prompt = " ".join(
            str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in prompt
        )
    seq_len = row.get("completion_tokens", row.get("num_completion_tokens", row.get("sequence_length", 0)))
    return {
        "trajectory_id": str(row.get("trajectory_id", f"{candidate_id}-t{index}")),
        "prompt_id": str(row.get("prompt_id", row.get("id", f"{candidate_id}-p{index}"))),
        "subject": str(row.get("subject", "unknown")),
        "difficulty": str(row.get("difficulty", row.get("level", "unknown"))),
        "prompt_text": str(prompt or ""),
        "completion_text": str(completion or ""),
        "reward_total": _scalar_reward(row.get("reward", row.get("score", row.get("reward_total", 0.0)))),
        "reward_exact_answer": _scalar_reward(
            metrics.get("boxed_answer_reward", metrics.get("reward", 0.0))
        ),
        "reward_format": _scalar_reward(metrics.get("format_reward", metrics.get("format", 0.0))),
        "advantage": _scalar_reward(row.get("advantage", 0.0)),
        "sequence_length": int(seq_len or 0),
    }


def read_rollouts(branch_output: Path, candidate_id: str) -> list[dict[str, Any]]:
    """Read + map all rollout rows prime-rl persisted for one branch run."""
    rows: list[dict[str, Any]] = []
    for path in _rollout_jsonls(branch_output):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return [_map_rollout_row(row, i, candidate_id) for i, row in enumerate(rows)]


def _optimizer_moment_norms(checkpoint: Path) -> tuple[float, float]:
    """Best-effort Adam first/second moment norms; 0.0 fallback (features-safe)."""
    if os.environ.get("TAP_NO_TORCH") == "1":
        return 0.0, 0.0
    try:
        import torch

        for name in ("optimizer.pt", "optimizer_state.pt", "optim.pt"):
            path = checkpoint / name
            if path.exists():
                state = torch.load(path, map_location="cpu", weights_only=False)
                first = second = 0.0
                packed = state.get("state", state) if isinstance(state, dict) else {}
                for entry in (packed.values() if isinstance(packed, dict) else []):
                    if isinstance(entry, dict):
                        if "exp_avg" in entry:
                            first += float(entry["exp_avg"].float().norm().item())
                        if "exp_avg_sq" in entry:
                            second += float(entry["exp_avg_sq"].float().norm().item())
                return first, second
    except Exception:
        pass
    return 0.0, 0.0


# --- TAP v1 collection driver (real GPU run via Mark's weights-only branch) ----

def _write_branch_config(
    config_path: Path, *, output_dir: Path, split_path: Path, model_name: str,
    renderer: str, run_name: str, args: argparse.Namespace,
) -> Path:
    """One prime-rl config for a state-gen warmup or a weights-only branch."""
    return write_prime_rl_config(
        config_path,
        PrimeRLConfigSpec(
            output_dir=output_dir,
            split_path=split_path,
            max_steps=1,
            batch_size=args.prompts_per_candidate,
            group_size=args.completions_per_prompt,
            seq_len=args.seq_len,
            max_completion_tokens=args.max_completion_tokens,
            model_name=model_name,
            lora_rank=args.lora_rank,
            learning_rate=args.learning_rate,
            gpus_per_node=max(args.gpu_count, 2),
            run_name=run_name,
            renderer_name=renderer,
        ),
    )


def run_controller(args: argparse.Namespace) -> None:
    """Real (Wave 2) TAP collection. Drives prime-rl with Mark's proven
    fresh-branch-weights recipe (weights-only branch: model_name=state ckpt,
    max_steps=1, renderer=default, run_default checkpoint layout) and writes the
    raw-artifact tree that math_loop.features converts to the 4 Parquet files."""
    from math_loop.controller import (
        checkpoint_steps, rl_command, run_command, weights_path,
    )
    from math_loop.data import build_probe_sets, prepare_training_splits, read_jsonl, write_jsonl
    from math_loop.tap_probes import (
        compute_policy_fingerprint, generic_incremental_kl, teacher_forced_probe_nll,
    )

    output_dir = Path(args.output_dir) / args.run_id
    raw_root = Path(args.raw_root)
    config_dir = Path(args.config_dir)
    log_dir = output_dir / "logs"
    raw_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[tap] collection start: {args.chains} chains x {args.states_per_chain} states x "
        f"{args.candidates_per_state} candidates; raw_root={raw_root}",
        flush=True,
    )
    training = prepare_training_splits(Path(args.data_dir), seed=args.seed)
    train_rows = read_jsonl(training.train_pool)
    probe_rows = read_jsonl(training.probe)
    by_id = {row["id"]: row for row in train_rows}
    prompt_ids = [row["id"] for row in train_rows]
    probes = build_probe_sets(probe_rows, seed=args.seed)

    schedule = build_tap_schedule(
        prompt_ids,
        chains=args.chains,
        states_per_chain=args.states_per_chain,
        candidates_per_state=args.candidates_per_state,
        prompts_per_candidate=args.prompts_per_candidate,
        seed=args.seed,
    )
    by_state: dict[tuple[int, int], list[TapCandidate]] = {}
    for cand in schedule:
        by_state.setdefault((cand.chain_index, cand.state_index), []).append(cand)

    for chain in range(args.chains):
        history: list[dict[str, Any]] = []
        state_ckpt: Path | None = None
        for state_index in range(args.states_per_chain):
            state_id = f"{chain}-{state_index}"
            if state_ckpt is None:
                state_out = output_dir / "states" / f"chain_{chain}"
                state_cfg = _write_branch_config(
                    config_dir / "states" / f"chain_{chain}.toml",
                    output_dir=state_out, split_path=Path(training.train_pool),
                    model_name=args.model_name, renderer="auto",
                    run_name=f"tap-states-{chain}", args=args,
                )
                if not checkpoint_steps(state_out):
                    print(f"[tap] generating initial state for chain {chain}...", flush=True)
                    run_command(rl_command(args, state_cfg),
                                log_path=log_dir / "states" / f"chain_{chain}.log")
                steps = checkpoint_steps(state_out)
                if not steps:
                    raise RuntimeError(f"state generation produced no checkpoint under {state_out}")
                state_ckpt = weights_path(state_out, max(steps))
            print(f"[tap] state {state_id}: before-state checkpoint {state_ckpt}", flush=True)

            matched_before = teacher_forced_probe_nll(state_ckpt, probes.matched)
            global_before = teacher_forced_probe_nll(state_ckpt, probes.global_probe)
            generic_before = 0.0
            fingerprint = compute_policy_fingerprint(state_ckpt, probes.fingerprint)
            probe_before = {
                "matched_probe_nll": matched_before,
                "global_probe_nll": global_before,
                "generic_kl": generic_before,
            }

            before_d = branch.before_dir(raw_root, state_id)
            before_d.mkdir(parents=True, exist_ok=True)
            hashes = branch.compute_before_state_hashes(state_ckpt)
            (before_d / "hashes.json").write_text(json.dumps(hashes, sort_keys=True), encoding="utf-8")

            selected = select_main_chain_candidate(
                chain, state_index, args.candidates_per_state, args.seed
            )
            branch_ckpts: dict[int, Path] = {}
            for cand in by_state[(chain, state_index)]:
                k = cand.candidate_index
                prompt_rows = [by_id[pid] for pid in cand.prompt_ids if pid in by_id]
                branch_out = output_dir / "branches" / f"state_{state_id}" / f"cand_{k}"
                branch_split = branch_out / "candidate_prompts.jsonl"
                write_jsonl(branch_split, prompt_rows)
                branch_cfg = _write_branch_config(
                    config_dir / "branches" / f"{state_id}_cand_{k}.toml",
                    output_dir=branch_out, split_path=branch_split,
                    model_name=str(state_ckpt), renderer="default",
                    run_name=f"tap-{cand.candidate_id}", args=args,
                )
                print(f"[tap] branch {cand.candidate_id} (weights-only from {state_ckpt})", flush=True)
                run_command(rl_command(args, branch_cfg),
                            log_path=log_dir / "branches" / f"{cand.candidate_id}.log")
                bsteps = checkpoint_steps(branch_out)
                branch_ckpt = weights_path(branch_out, max(bsteps) if bsteps else 1)

                rollouts = read_rollouts(branch_out, cand.candidate_id)
                matched_after = teacher_forced_probe_nll(branch_ckpt, probes.matched)
                global_after = teacher_forced_probe_nll(branch_ckpt, probes.global_probe)
                generic_after = generic_incremental_kl(state_ckpt, branch_ckpt, probes.generic_drift)
                grad_sketch = branch.compute_lora_gradient_sketch(branch_ckpt, prompt_rows)
                branch.write_candidate_artifacts(
                    branch.candidate_dir(raw_root, state_id, k),
                    rollouts=rollouts,
                    probe_before=probe_before,
                    probe_after={
                        "matched_probe_nll": matched_after,
                        "global_probe_nll": global_after,
                        "generic_kl": generic_after,
                    },
                    grad_sketch=grad_sketch,
                )
                branch_ckpts[k] = branch_ckpt
                print(f"[tap] labeled {cand.candidate_id} ({len(rollouts)} rollouts)", flush=True)

            adam_first, adam_second = _optimizer_moment_norms(state_ckpt)
            state_json = {
                "state_id": state_id,
                "chain_id": str(chain),
                "step": state_index,
                "seed": args.seed,
                "learning_rate": args.learning_rate,
                "grpo_beta": args.grpo_beta,
                "clip_range": 0.0,
                "lora_rank": args.lora_rank,
                "matched_probe_nll_before": matched_before,
                "global_probe_nll_before": global_before,
                "generic_kl_before": generic_before,
                "adam_first_moment_norm": adam_first,
                "adam_second_moment_norm": adam_second,
                "policy_fingerprint": fingerprint,
                "history": list(history),
                "selected_candidate_index": selected,
            }
            state_dir = branch.state_dir(raw_root, state_id)
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "state.json").write_text(json.dumps(state_json, sort_keys=True), encoding="utf-8")

            # Advance the main chain with the SEEDED-RANDOM candidate (spec).
            state_ckpt = branch_ckpts.get(selected, state_ckpt)
            history.append({
                "historical_candidate_id": f"{state_id}-{selected}",
                "candidate_id": f"{state_id}-{selected}",
            })
            history = history[-HISTORY_WINDOW:]

    print(f"[tap] collection complete; raw artifacts under {raw_root}", flush=True)


# --- CLI ---------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="tap_v1")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tap"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--config-dir", type=Path, default=Path("outputs/tap/configs"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/math_loop"))
    parser.add_argument("--rl-command", default="uv run rl")
    # 2 chains x 6 states x 6 candidates (spec compressed scale). Scale 48..128.
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--states-per-chain", type=int, default=6)
    parser.add_argument("--candidates-per-state", type=int, default=6)
    parser.add_argument("--prompts-per-candidate", type=int, default=2)
    parser.add_argument("--completions-per-prompt", type=int, default=4)
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--grpo-beta", type=float, default=0.04)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--max-completion-tokens", type=int, default=192)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.raw_root is None:
        args.raw_root = args.output_dir / args.run_id / "raw"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        plan = build_plan(args)
        print(render_plan(plan, args))
        return 0
    run_controller(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
