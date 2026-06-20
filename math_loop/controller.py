"""Controller for 48-state x 16-candidate prime-rl branch labeling."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any

from math_loop.data import prepare_training_splits, read_jsonl, write_jsonl
from math_loop.prime_rl_config import PrimeRLConfigSpec, write_prime_rl_config
from math_loop.probe_loss import compute_probe_loss
from math_loop.schedule import CandidateBatch, build_candidate_schedule


CONTROLLER_VERSION = "fresh-branch-weights-v1"


def run_command(command: list[str], *, dry_run: bool = False, log_path: Path | None = None) -> None:
    rendered = shlex.join(command)
    if dry_run:
        print(rendered)
        return
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"$ {rendered}\n")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
            returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
        return
    subprocess.run(command, check=True)


def checkpoint_steps(output_dir: Path) -> list[int]:
    steps = set()
    for ckpt_dir in checkpoint_roots(output_dir):
        for path in ckpt_dir.glob("step_*"):
            try:
                steps.add(int(path.name.rsplit("_", 1)[1]))
            except ValueError:
                continue
    return sorted(steps)


def checkpoint_roots(output_dir: Path) -> list[Path]:
    return [output_dir / "run_default" / "checkpoints", output_dir / "checkpoints"]


def weight_roots(output_dir: Path) -> list[Path]:
    return [output_dir / "run_default" / "weights", output_dir / "weights"]


def checkpoint_path(output_dir: Path, step: int) -> Path:
    for ckpt_dir in checkpoint_roots(output_dir):
        candidate = ckpt_dir / f"step_{step}"
        if candidate.exists():
            return candidate
    return checkpoint_roots(output_dir)[0] / f"step_{step}"


def weights_path(output_dir: Path, step: int) -> Path:
    for weights_dir in weight_roots(output_dir):
        candidate = weights_dir / f"step_{step}"
        if candidate.exists():
            return candidate
    return checkpoint_path(output_dir, step)


def select_rows(rows: list[dict[str, Any]], prompt_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    by_id = {row["id"]: row for row in rows}
    missing = [prompt_id for prompt_id in prompt_ids if prompt_id not in by_id]
    if missing:
        raise KeyError(f"candidate references unknown prompt ids: {missing}")
    selected = []
    for prompt_id in prompt_ids:
        row = dict(by_id[prompt_id])
        row["candidate_prompt_id"] = prompt_id
        selected.append(row)
    return selected


def read_existing_label_keys(path: Path) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    for row in read_jsonl(path):
        keys.add((int(row["state_index"]), int(row["candidate_index"])))
    return keys


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def reward_summary(branch_output: Path) -> dict[str, float | int]:
    rewards: list[float] = []
    token_counts: list[float] = []
    rollout_roots = [branch_output / "rollouts", branch_output / "run_default" / "rollouts"]
    for root in rollout_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            _collect_rollout_summary(path, rewards, token_counts)
    summary: dict[str, float | int] = {}
    if rewards:
        summary["reward_count"] = len(rewards)
        summary["reward_mean"] = sum(rewards) / len(rewards)
    if token_counts:
        summary["completion_token_mean"] = sum(token_counts) / len(token_counts)
    return summary


def _collect_rollout_summary(path: Path, rewards: list[float], token_counts: list[float]) -> None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in ("reward", "score"):
                    value = row.get(key)
                    if isinstance(value, (int, float)):
                        rewards.append(float(value))
                        break
                for key in ("completion_tokens", "num_completion_tokens", "tokens"):
                    value = row.get(key)
                    if isinstance(value, (int, float)):
                        token_counts.append(float(value))
                        break
    except OSError:
        return


def write_run_config(
    path: Path,
    *,
    output_dir: Path,
    split_path: Path,
    max_steps: int,
    args: argparse.Namespace,
    clean_output_dir: bool,
    run_name: str,
    model_name: str | None = None,
    renderer_name: str = "auto",
) -> Path:
    return write_prime_rl_config(
        path,
        PrimeRLConfigSpec(
            output_dir=output_dir,
            split_path=split_path,
            max_steps=max_steps,
            batch_size=args.batch_prompts,
            group_size=args.group_size,
            seq_len=args.seq_len,
            max_completion_tokens=args.max_completion_tokens,
            model_name=model_name or args.model_name,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            learning_rate=args.learning_rate,
            gpus_per_node=max(args.gpu_count, 2),
            clean_output_dir=clean_output_dir,
            run_name=run_name,
            renderer_name=renderer_name,
        ),
    )


def rl_command(args: argparse.Namespace, config_path: Path, *, resume_step: int | None = None) -> list[str]:
    command = [*shlex.split(args.rl_command), "@", str(config_path), "--ckpt", "--ckpt.interval", "1"]
    if resume_step is not None:
        command.extend(["--ckpt.resume-step", str(resume_step)])
    return command


def maybe_probe(
    checkpoint: Path,
    probe_path: Path,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.skip_probe_loss:
        print(f"Skipping probe loss for {checkpoint}", flush=True)
        return {
            "checkpoint": str(checkpoint),
            "examples": 0,
            "tokens": 0,
            "nll": None,
        }
    result = compute_probe_loss(
        checkpoint,
        probe_path,
        model_name=args.model_name,
        batch_size=args.probe_batch_size,
        max_length=args.seq_len,
        device=args.probe_device,
        dtype=args.probe_dtype,
    )
    return asdict(result)


def run_controller(args: argparse.Namespace) -> None:
    print(
        f"Starting math loop controller ({CONTROLLER_VERSION}): states={args.states}, "
        f"candidates_per_state={args.candidates_per_state}, "
        f"batch_prompts={args.batch_prompts}, group_size={args.group_size}",
        flush=True,
    )
    if args.force and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = args.output_dir / "configs"
    log_dir = args.output_dir / "logs"
    label_path = args.labels_path or (args.output_dir / "labels.jsonl")

    print(f"Preparing MATH train/probe splits in {args.data_dir}...", flush=True)
    training = prepare_training_splits(
        args.data_dir,
        seed=args.seed,
        probe_size=args.probe_size,
        force=args.force_data,
    )
    print(f"Using train split {training.train_pool}", flush=True)
    print(f"Using probe split {training.probe}", flush=True)
    train_rows = read_jsonl(training.train_pool)
    prompt_ids = [row["id"] for row in train_rows]
    schedule = build_candidate_schedule(
        prompt_ids,
        states=args.states,
        candidates_per_state=args.candidates_per_state,
        batch_prompts=args.batch_prompts,
        seed=args.seed,
    )

    state_output = args.output_dir / "states"
    state_config = write_run_config(
        config_dir / "state_generation.toml",
        output_dir=state_output,
        split_path=training.train_pool,
        max_steps=args.states,
        args=args,
        clean_output_dir=args.force,
        run_name="qwen3-math-states",
    )
    steps = checkpoint_steps(state_output)
    if len(steps) < args.states:
        print(
            f"Launching prime-rl state generation for {args.states} step(s). "
            f"Logs: {log_dir / 'state_generation.log'}",
            flush=True,
        )
        run_command(
            rl_command(args, state_config),
            dry_run=args.dry_run,
            log_path=log_dir / "state_generation.log",
        )
        steps = checkpoint_steps(state_output)
        print(f"State generation finished; found checkpoint steps: {steps}", flush=True)

    if args.dry_run:
        print(f"would write labels to {label_path}")
        return
    if len(steps) < args.states:
        raise RuntimeError(f"expected {args.states} state checkpoints, found {len(steps)}")

    steps = steps[: args.states]
    schedule_by_state: dict[int, list[CandidateBatch]] = {}
    for candidate in schedule:
        schedule_by_state.setdefault(candidate.state_index, []).append(candidate)

    existing = read_existing_label_keys(label_path)
    state_probe_cache: dict[int, dict[str, Any]] = {}
    for ordinal, step in enumerate(steps, start=1):
        state_checkpoint = weights_path(state_output, step)
        print(f"Processing state {ordinal}/{len(steps)} from step {step}: {state_checkpoint}", flush=True)
        state_probe_cache[ordinal] = maybe_probe(state_checkpoint, training.probe, args=args)
        for candidate in schedule_by_state[ordinal]:
            key = (candidate.state_index, candidate.candidate_index)
            if key in existing and not args.force_labels:
                continue

            branch_output = (
                args.output_dir
                / "branches"
                / f"state_{candidate.state_index:03d}_step_{step}"
                / f"candidate_{candidate.candidate_index:02d}"
            )
            branch_split = branch_output / "candidate_prompts.jsonl"
            write_jsonl(branch_split, select_rows(train_rows, candidate.prompt_ids))
            print(
                f"Launching branch state={candidate.state_index} "
                f"candidate={candidate.candidate_index} prompts={list(candidate.prompt_ids)}. "
                f"Initial weights: {state_checkpoint}. "
                f"Logs: {log_dir / 'branches' / f'state_{candidate.state_index:03d}_candidate_{candidate.candidate_index:02d}.log'}",
                flush=True,
            )

            branch_config = write_run_config(
                config_dir
                / "branches"
                / f"state_{candidate.state_index:03d}_step_{step}"
                / f"candidate_{candidate.candidate_index:02d}.toml",
                output_dir=branch_output,
                split_path=branch_split,
                max_steps=1,
                args=args,
                clean_output_dir=False,
                run_name=f"qwen3-math-s{candidate.state_index:03d}-c{candidate.candidate_index:02d}",
                model_name=str(state_checkpoint),
                renderer_name="default",
            )
            run_command(
                rl_command(args, branch_config),
                log_path=log_dir
                / "branches"
                / f"state_{candidate.state_index:03d}_candidate_{candidate.candidate_index:02d}.log",
            )
            branch_checkpoint = weights_path(branch_output, 1)
            state_probe = state_probe_cache[ordinal]
            branch_probe = maybe_probe(branch_checkpoint, training.probe, args=args)
            state_nll = state_probe["nll"]
            branch_nll = branch_probe["nll"]
            delta_nll = None if state_nll is None or branch_nll is None else branch_nll - state_nll
            append_jsonl(
                label_path,
                {
                    "state_index": candidate.state_index,
                    "state_step": step,
                    "candidate_index": candidate.candidate_index,
                    "prompt_ids": list(candidate.prompt_ids),
                    "state_checkpoint": str(state_checkpoint),
                    "branch_checkpoint": str(branch_checkpoint),
                    "state_probe": state_probe,
                    "branch_probe": branch_probe,
                    "delta_probe_nll": delta_nll,
                    "reward_summary": reward_summary(branch_output),
                },
            )
            print(f"Wrote label for state={candidate.state_index} candidate={candidate.candidate_index}", flush=True)
            existing.add(key)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/prime_rl/qwen3_8b_math_state.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen3_math_loop"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/math_loop"))
    parser.add_argument("--labels-path", type=Path)
    parser.add_argument("--rl-command", default="uv run rl")
    parser.add_argument("--states", type=int, default=48)
    parser.add_argument("--candidates-per-state", type=int, default=16)
    parser.add_argument("--batch-prompts", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--probe-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--gpu-count", type=int, default=2)
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--probe-batch-size", type=int, default=1)
    parser.add_argument("--probe-device", default="cuda")
    parser.add_argument("--probe-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--skip-probe-loss", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--force-labels", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.config.exists():
        raise SystemExit(f"config does not exist: {args.config}")
    run_controller(args)


if __name__ == "__main__":
    main()
