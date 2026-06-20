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


# --- branch worker (Wave 2; picklable for ProcessPoolExecutor) ---------------

def _branch_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Run ONE candidate branch on a pinned worker GPU and write raw artifacts.

    Pins ``CUDA_VISIBLE_DEVICES`` before any torch import so the subprocess and
    in-process probe evaluation land on the assigned GPU. Heavy imports are
    deferred to keep this module importable on CPU.
    """
    gpu = task.get("gpu")
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    from math_loop.controller import copy_resume_checkpoint, run_command, weights_path
    from math_loop.data import write_jsonl

    args = task["args"]
    candidate = task["candidate"]
    step = task["step"]
    state_output = Path(task["state_output"])
    raw_root = Path(task["raw_root"])

    branch_output = Path(task["branch_output"])
    branch_split = branch_output / "candidate_prompts.jsonl"
    write_jsonl(branch_split, task["prompt_rows"])
    copy_resume_checkpoint(state_output, branch_output, step)

    config_path = write_prime_rl_config(
        Path(task["config_path"]),
        PrimeRLConfigSpec(
            output_dir=branch_output,
            split_path=branch_split,
            max_steps=step + 1,
            batch_size=args["prompts_per_candidate"],
            group_size=args["completions_per_prompt"],
            seq_len=args["seq_len"],
            max_completion_tokens=args["max_completion_tokens"],
            model_name=args["model_name"],
            lora_rank=args["lora_rank"],
            learning_rate=args["learning_rate"],
            gpus_per_node=2,
            run_name=f"tap-{candidate['candidate_id']}",
        ),
    )
    run_command(branch.branch_command(args["rl_command"], config_path, resume_step=step))

    branch_checkpoint = weights_path(branch_output, step + 1)
    cand_dir = branch.candidate_dir(raw_root, candidate["state_id"], candidate["candidate_index"])

    from math_loop.tap_probes import teacher_forced_probe_nll

    probe_after = {
        "matched_probe_nll": teacher_forced_probe_nll(branch_checkpoint, task["matched_probe"]),
        "global_probe_nll": teacher_forced_probe_nll(branch_checkpoint, task["global_probe"]),
        "generic_kl": task["generic_kl_after"],
    }
    grad_sketch = branch.compute_lora_gradient_sketch(branch_checkpoint, task["prompt_rows"])
    branch.write_candidate_artifacts(
        cand_dir,
        rollouts=task["rollouts"],
        probe_before=task["probe_before"],
        probe_after=probe_after,
        grad_sketch=grad_sketch,
    )
    return {"candidate_id": candidate["candidate_id"], "cand_dir": str(cand_dir)}


def run_controller(args: argparse.Namespace) -> None:
    """Real (Wave 2) collection run. Requires the prime-rl GPU environment."""
    raise SystemExit(
        "tap_controller real run executes on the Wave 2 GPU pod (prime-rl + H100s). "
        "Use --dry-run for Wave 1 CPU validation."
    )


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
