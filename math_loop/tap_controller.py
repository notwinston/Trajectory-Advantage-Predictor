"""TAP v1 data-collection controller: 3 chains x 8 states x 8 candidates.

Drives the spec's compressed collection loop (section "DATA-COLLECTION LOOP"):

* For each chain, walk 8 policy states. At each state save the before-state
  (adapter + optimizer + step/seed/lr/grpo_beta), measure the probes, and record
  the 16-value policy fingerprint.
* Generate 8 candidate GRPO batches (2 prompts x 4 completions = 8 trajectories
  each), branch every candidate from the byte-identical before-state, label it,
  and write the raw-artifact tree :mod:`math_loop.features` consumes.
* After all 8 candidates are labeled, advance the main chain with a
  SEEDED-RANDOM candidate (keeps the collected history unbiased) and keep the
  last 8 applied updates as history.

Each state is processed in three phases to avoid redundant GPU work: PHASE A runs
the 8 branch GRPO subprocesses (prime-rl), PHASE B loads ONE resident base model
(``ProbeSession``) and labels all 8 candidates via cheap LoRA-adapter swaps, and
PHASE C advances the chain + prunes. A PERSISTENT vLLM inference server
(``--persistent-inference``, GPU 1) is booted once and reused by every branch so
each GRPO step skips the ~80-90s cold boot; the per-branch ``uv run rl`` omits its
own ``[inference]`` block and connects to that server (GPU 0 for the trainer),
falling back to per-branch inference if the server never becomes ready. All
GPU/prime-rl execution is Wave 2 — ``--dry-run`` only plans, makes NO network
calls, and imports nothing heavy. Module-level imports are stdlib + pure
``math_loop`` modules so ``import math_loop.tap_controller`` works with
torch/transformers/prime absent.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from math_loop import branch
from math_loop.prime_rl_config import PrimeRLConfigSpec, write_prime_rl_config
from math_loop.schedule import TapCandidate, build_tap_schedule

TRAJECTORIES_PER_CANDIDATE = 8  # 2 prompts x 4 completions
HISTORY_WINDOW = 8


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


def cleanup_cuda_if_available(device: str = "cuda") -> None:
    """Best-effort CUDA cache cleanup between local probe/sketch work and vLLM."""
    try:
        import torch
    except Exception:
        return
    try:
        from math_loop.probe_loss import _cleanup_torch_cuda

        _cleanup_torch_cuda(torch, device)
    except Exception:
        return


def prune_checkpoint_weights(checkpoint: Path, output_dir: Path) -> bool:
    """Delete the containing weights directory for a checkpoint inside output_dir."""
    try:
        resolved_output = output_dir.resolve()
        resolved_checkpoint = checkpoint.resolve()
    except OSError:
        return False
    if resolved_output not in (resolved_checkpoint, *resolved_checkpoint.parents):
        return False
    weights_dir: Path | None = None
    for candidate in (resolved_checkpoint, *resolved_checkpoint.parents):
        if candidate.name == "weights":
            weights_dir = candidate
            break
    if weights_dir is None or not weights_dir.exists():
        return False
    shutil.rmtree(weights_dir)
    return True


def prune_prime_rl_checkpoints(run_dir: Path, output_dir: Path) -> bool:
    """Delete bulky prime-rl checkpoint trees for a completed branch/state run."""

    try:
        resolved_output = output_dir.resolve()
        resolved_run = run_dir.resolve()
    except OSError:
        return False
    if resolved_output not in (resolved_run, *resolved_run.parents):
        return False
    removed = False
    for checkpoint_dir in (
        resolved_run / "checkpoints",
        resolved_run / "run_default" / "checkpoints",
    ):
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            removed = True
    return removed


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


def _manifest_path(raw_root: str | Path, chain_index: int, state_index: int) -> Path:
    return Path(raw_root) / "_manifests" / f"chain_{chain_index}" / f"state_{state_index}.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_chain_state_manifest(
    raw_root: str | Path,
    *,
    chain_index: int,
    completed_state_index: int,
    state_ckpt: Path,
    history: list[dict[str, Any]],
) -> None:
    """Persist the ready-to-run next state after a completed state.

    This mirrors Vincent's state-manifest resume idea, but stores only Winston's
    production controller state: checkpoint path plus the compact selected
    history. If a later process sees the manifest for ``state_index + 1``, the
    prior state was fully labeled and can be skipped.
    """
    next_state_index = completed_state_index + 1
    _atomic_write_json(
        _manifest_path(raw_root, chain_index, next_state_index),
        {
            "chain_index": chain_index,
            "completed_state_index": completed_state_index,
            "ready_state_index": next_state_index,
            "state_ckpt": str(state_ckpt),
            "history": history[-HISTORY_WINDOW:],
        },
    )


def load_chain_state_manifest(
    raw_root: str | Path, chain_index: int, state_index: int
) -> dict[str, Any] | None:
    path = _manifest_path(raw_root, chain_index, state_index)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
    lines.append(
        "dispatch: GPU 1 = persistent vLLM inference server (booted once, reused by "
        "every branch); GPU 0 = per-branch trainer (Phase A) + resident probe model "
        "(Phase B)"
    )
    lines.append(f"already labeled (resume-safe skip): {len(plan.already_labeled)}/{plan.total_labels}")
    lines.append("")

    for chain_index in range(plan.chains):
        lines.append(f"chain {chain_index}:")
        for state_index in range(plan.states_per_chain):
            state_id = f"{chain_index}-{state_index}"
            chosen = plan.selected_per_state[state_id]
            lines.append(
                f"  state {state_id}: branch candidates 0..{plan.candidates_per_state - 1} "
                f"-> label all (1 resident base) -> apply seeded-random candidate cand_{chosen}"
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


def _rollout_sequence_length(row: dict[str, Any], completion: Any) -> int:
    for key in (
        "completion_tokens",
        "num_completion_tokens",
        "sequence_length",
        "completion_token_count",
    ):
        try:
            value = int(row.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    for key in ("completion_token_ids", "token_ids", "tokens", "token_logprobs"):
        value = row.get(key)
        if isinstance(value, list) and value:
            return len(value)
    text = str(completion or "")
    if not text:
        return 0
    return max(1, len(text.split()), len(text) // 4)


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
    seq_len = _rollout_sequence_length(row, completion)
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
        "sequence_length": seq_len,
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
    renderer: str, run_name: str, args: argparse.Namespace, persistent: bool = False,
) -> Path:
    """One prime-rl config for a state-gen warmup or a weights-only branch.

    With ``persistent=True`` the ``[inference]`` block is omitted (deployment is
    trainer-only) so ``uv run rl`` connects to the long-lived external inference
    server instead of cold-booting its own vLLM per step.
    """
    return write_prime_rl_config(
        config_path,
        PrimeRLConfigSpec(
            output_dir=output_dir,
            split_path=split_path,
            max_steps=1,
            # prime-rl's orchestrator batch_size is total rollout samples, and
            # must be divisible by env group_size / samples per problem.
            batch_size=args.prompts_per_candidate * args.completions_per_prompt,
            group_size=args.completions_per_prompt,
            seq_len=args.seq_len,
            max_completion_tokens=args.max_completion_tokens,
            model_name=model_name,
            lora_rank=args.lora_rank,
            learning_rate=args.learning_rate,
            gpus_per_node=1 if persistent else max(args.gpu_count, 2),
            num_infer_gpus=0 if persistent else 1,
            num_train_gpus=1,
            include_inference=not persistent,
            inference_gpu_memory_utilization=getattr(args, "inference_gpu_memory_utilization", 0.80),
            run_name=run_name,
            renderer_name=renderer,
        ),
    )


def _maybe_start_inference_server(
    args: argparse.Namespace, *, config_dir: Path, log_dir: Path, train_pool: Path
):
    """Start the persistent vLLM inference server (GPU 1) when --persistent-inference.

    Returns ``(persistent, proc, branch_env)``. On readiness timeout it tears the
    server down and returns ``persistent=False`` so the caller transparently falls
    back to the per-branch (cold-boot) inference path.
    """
    if not getattr(args, "persistent_inference", False):
        return False, None, None
    from math_loop.controller import start_background_process, stop_process, wait_for_http_ready
    from math_loop.prime_rl_config import PrimeRLConfigSpec, write_inference_server_config

    infer_cfg = write_inference_server_config(
        config_dir / "inference.toml",
        PrimeRLConfigSpec(
            output_dir=Path(args.output_dir) / args.run_id,
            split_path=train_pool,
            max_steps=1,
            model_name=args.model_name,
            lora_rank=args.lora_rank,
            max_loras=max(args.candidates_per_state, 8),
            inference_gpu_memory_utilization=args.inference_gpu_memory_utilization,
        ),
    )
    # "uv run rl" -> "uv run inference"
    infer_cmd = [*shlex.split(args.rl_command)[:-1], "inference", "@", str(infer_cfg)]
    print(f"[tap] starting persistent inference server (GPU 1): {shlex.join(infer_cmd)}", flush=True)
    proc = start_background_process(
        infer_cmd,
        env_overrides={"CUDA_VISIBLE_DEVICES": "1"},
        log_path=log_dir / "inference_server.log",
    )
    url = "http://127.0.0.1:8000/health"
    print(f"[tap] waiting for inference server at {url} (timeout {args.inference_ready_timeout}s)...", flush=True)
    if wait_for_http_ready(url, timeout=args.inference_ready_timeout, proc=proc):
        print("[tap] persistent inference server ready; branches reuse it (no per-step cold boot)", flush=True)
        return True, proc, {"CUDA_VISIBLE_DEVICES": "0"}
    print("[tap] WARNING: inference server not ready; falling back to per-branch inference", flush=True)
    stop_process(proc)
    return False, None, None


def _stop_inference_server(proc) -> None:
    if proc is None:
        return
    from math_loop.controller import stop_process

    stop_process(proc)
    print("[tap] stopped persistent inference server", flush=True)


def run_controller(args: argparse.Namespace) -> None:
    """Real (Wave 2) TAP collection. Drives prime-rl with Mark's proven
    fresh-branch-weights recipe (weights-only branch: model_name=state ckpt,
    max_steps=1, renderer=default, run_default checkpoint layout) and writes the
    raw-artifact tree that math_loop.features converts to the 4 Parquet files."""
    from math_loop.controller import (
        checkpoint_steps, rl_command, run_command, weights_path,
    )
    from math_loop.data import build_probe_sets, prepare_training_splits, read_jsonl, write_jsonl
    from math_loop.probe_loss import ProbeSession

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

    # Persistent inference server (GPU 1): booted lazily and reused across a chain's
    # branches so each GRPO step skips the ~80-90s vLLM cold boot. It is RESTARTED
    # (back to the base model) before each chain's state generation -- state-gen
    # trains from scratch and pushes no weights, so a stale LoRA left by a prior
    # chain's branches would otherwise pollute the new state-0 rollouts. Branch runs
    # self-correct (the orchestrator pushes the resumed before-state LoRA on startup),
    # so they only need the server up. ``branch_env`` pins those rl runs to GPU 0.
    import atexit

    persistent_requested = bool(getattr(args, "persistent_inference", False))
    server: dict[str, Any] = {"proc": None}  # mutable so atexit sees the live process
    persistent = False
    branch_env: dict[str, str] | None = None
    if persistent_requested:
        atexit.register(lambda: _stop_inference_server(server["proc"]))

    for chain in range(args.chains):
        history: list[dict[str, Any]] = []
        state_ckpt: Path | None = None
        for state_index in range(args.states_per_chain):
            state_id = f"{chain}-{state_index}"
            resume_manifest = load_chain_state_manifest(raw_root, chain, state_index)
            if resume_manifest is not None:
                state_ckpt = Path(resume_manifest["state_ckpt"])
                history = list(resume_manifest.get("history", []))

            completed_manifest = load_chain_state_manifest(raw_root, chain, state_index + 1)
            if completed_manifest is not None:
                state_ckpt = Path(completed_manifest["state_ckpt"])
                history = list(completed_manifest.get("history", []))
                print(f"[tap] state {state_id}: already complete; resuming at next state", flush=True)
                continue

            # Manage the persistent inference server for this state: restart it to the
            # base model before a state generation, otherwise just ensure it is up.
            if persistent_requested:
                if state_ckpt is None and server["proc"] is not None:
                    _stop_inference_server(server["proc"])
                    server["proc"] = None
                if server["proc"] is None:
                    persistent, server["proc"], branch_env = _maybe_start_inference_server(
                        args, config_dir=config_dir, log_dir=log_dir,
                        train_pool=Path(training.train_pool),
                    )
                    if server["proc"] is None:
                        persistent_requested = False  # start failed -> per-branch fallback

            if state_ckpt is None:
                state_out = output_dir / "states" / f"chain_{chain}"
                state_cfg = _write_branch_config(
                    config_dir / "states" / f"chain_{chain}.toml",
                    output_dir=state_out, split_path=Path(training.train_pool),
                    model_name=args.model_name, renderer="auto",
                    run_name=f"tap-states-{chain}", args=args, persistent=persistent,
                )
                if not checkpoint_steps(state_out):
                    print(f"[tap] generating initial state for chain {chain}...", flush=True)
                    run_command(rl_command(args, state_cfg),
                                log_path=log_dir / "states" / f"chain_{chain}.log",
                                env_overrides=branch_env)
                steps = checkpoint_steps(state_out)
                if not steps:
                    raise RuntimeError(f"state generation produced no checkpoint under {state_out}")
                state_ckpt = weights_path(state_out, max(steps))
            print(f"[tap] state {state_id}: before-state checkpoint {state_ckpt}", flush=True)

            # === PHASE A: branch subprocesses (NO resident probe model; both GPUs
            # free for prime-rl). Collect each candidate's branch checkpoint +
            # rollouts. Defer ALL branch-weight pruning to Phase C so the probe pass
            # can read every adapter; only the bulky trainer checkpoints prune here.
            before_d = branch.before_dir(raw_root, state_id)
            before_d.mkdir(parents=True, exist_ok=True)
            hashes = branch.compute_before_state_hashes(state_ckpt)
            (before_d / "hashes.json").write_text(json.dumps(hashes, sort_keys=True), encoding="utf-8")

            selected = select_main_chain_candidate(
                chain, state_index, args.candidates_per_state, args.seed
            )
            previous_state_ckpt = state_ckpt
            state_candidates = by_state[(chain, state_index)]
            branch_ckpts: dict[int, Path] = {}
            rollouts_by_k: dict[int, list[dict[str, Any]]] = {}
            prompt_rows_by_k: dict[int, list[dict[str, Any]]] = {}
            for cand in state_candidates:
                k = cand.candidate_index
                prompt_rows = [by_id[pid] for pid in cand.prompt_ids if pid in by_id]
                prompt_rows_by_k[k] = prompt_rows
                branch_out = output_dir / "branches" / f"state_{state_id}" / f"cand_{k}"
                branch_split = branch_out / "candidate_prompts.jsonl"
                write_jsonl(branch_split, prompt_rows)
                branch_cfg = _write_branch_config(
                    config_dir / "branches" / f"{state_id}_cand_{k}.toml",
                    output_dir=branch_out, split_path=branch_split,
                    model_name=str(state_ckpt), renderer="default",
                    run_name=f"tap-{cand.candidate_id}", args=args, persistent=persistent,
                )
                print(f"[tap] branch {cand.candidate_id} (weights-only from {state_ckpt})", flush=True)
                cleanup_cuda_if_available()
                run_command(rl_command(args, branch_cfg),
                            log_path=log_dir / "branches" / f"{cand.candidate_id}.log",
                            env_overrides=branch_env)
                bsteps = checkpoint_steps(branch_out)
                branch_ckpts[k] = weights_path(branch_out, max(bsteps) if bsteps else 1)
                rollouts_by_k[k] = read_rollouts(branch_out, cand.candidate_id)
                if prune_prime_rl_checkpoints(branch_out, output_dir):
                    print(f"[tap] pruned branch trainer checkpoints for {cand.candidate_id}", flush=True)

            # === PHASE B: probe pass with ONE resident base. The state adapter is
            # loaded once for the before-probes + fingerprint + cached generic-KL
            # base side (identical across the state's candidates); each candidate's
            # after-probes/KL/grad-sketch reuse the resident base via adapter swaps.
            session = ProbeSession(
                model_name=args.model_name, device="cuda:0",
                dtype="bfloat16", max_length=args.seq_len,
            )
            try:
                session.use_adapter(state_ckpt)
                matched_before = session.probe_nll(probes.matched, batch_size=args.probe_batch_size)
                global_before = session.probe_nll(probes.global_probe, batch_size=args.probe_batch_size)
                generic_before = 0.0
                fingerprint = session.fingerprint(probes.fingerprint)
                generic_base_cache = session.logits_for_generic(probes.generic_drift)
                probe_before = {
                    "matched_probe_nll": matched_before,
                    "global_probe_nll": global_before,
                    "generic_kl": generic_before,
                }
                for cand in state_candidates:
                    k = cand.candidate_index
                    session.use_adapter(branch_ckpts[k])
                    matched_after = session.probe_nll(probes.matched, batch_size=args.probe_batch_size)
                    global_after = session.probe_nll(probes.global_probe, batch_size=args.probe_batch_size)
                    generic_after = session.sequence_kl_vs(generic_base_cache)
                    grad_sketch = session.grad_sketch(prompt_rows_by_k[k])
                    rollouts = rollouts_by_k.get(k, [])
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
                    print(f"[tap] labeled {cand.candidate_id} ({len(rollouts)} rollouts)", flush=True)
            finally:
                session.close()
                cleanup_cuda_if_available()

            # === PHASE C: state.json + seeded-random advance + now-safe pruning.
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

            # Prune non-selected branch weights now that the probe pass is done
            # (deferred from Phase A so every adapter was readable during probing).
            for k, branch_ckpt in branch_ckpts.items():
                if k != selected and prune_checkpoint_weights(branch_ckpt, output_dir):
                    print(f"[tap] pruned non-selected branch weights for {state_id}-{k}", flush=True)

            # Advance the main chain with the SEEDED-RANDOM candidate (spec).
            state_ckpt = branch_ckpts.get(selected, state_ckpt)
            if state_ckpt != previous_state_ckpt and prune_checkpoint_weights(previous_state_ckpt, output_dir):
                print(f"[tap] pruned superseded state weights for {state_id}", flush=True)
            history.append({
                "historical_candidate_id": f"{state_id}-{selected}",
                "candidate_id": f"{state_id}-{selected}",
            })
            history = history[-HISTORY_WINDOW:]
            write_chain_state_manifest(
                raw_root,
                chain_index=chain,
                completed_state_index=state_index,
                state_ckpt=state_ckpt,
                history=history,
            )
        if state_ckpt is not None and prune_checkpoint_weights(state_ckpt, output_dir):
            print(f"[tap] pruned final checkpoint weights for chain {chain}", flush=True)

    _stop_inference_server(server["proc"])
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
    parser.add_argument("--chains", type=int, default=3)
    parser.add_argument("--states-per-chain", type=int, default=8)
    parser.add_argument("--candidates-per-state", type=int, default=8)
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
    parser.add_argument("--probe-batch-size", type=int, default=8,
                        help="batch size for teacher-forced probe NLL forwards (Phase B)")
    parser.add_argument("--persistent-inference", action=argparse.BooleanOptionalAction, default=True,
                        help="reuse ONE vLLM inference server across all branches "
                             "(skips the ~80-90s per-step cold boot); falls back to "
                             "per-branch inference if the server is unreachable")
    parser.add_argument("--inference-gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--inference-ready-timeout", type=float, default=1800.0)
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
