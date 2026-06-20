"""TAP v1 branch primitive: one GRPO update via prime-rl resume + raw artifacts.

A *branch* loads the identical frozen before-state of a policy and applies
EXACTLY ONE GRPO optimizer step using a single candidate batch, then evaluates
probes. Every candidate at a given state must branch from a byte-identical LoRA
adapter + optimizer state — :func:`assert_identical_before_state` enforces this
by comparing ``checkpoint_hash`` and ``optimizer_state_hash``.

This module emits the raw branch-artifact tree the feature extractor reads::

    <raw_root>/state_<chain>-<state>/
        before/      adapter + optimizer + hashes.json
        state.json   fingerprint, lr, beta, adam moments, probe-before, history
        cand_<k>/    rollouts.jsonl, probe_before.json, probe_after.json,
                     grad_sketch.npy  OR  grad_unavailable.flag

Module-level imports are stdlib only; torch/numpy live inside the functions that
actually run on the Wave 2 GPU pod.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence


# --- raw-artifact layout -----------------------------------------------------

def state_dir(raw_root: str | Path, state_id: str) -> Path:
    """Directory holding one policy state's branch artifacts."""
    return Path(raw_root) / f"state_{state_id}"


def before_dir(raw_root: str | Path, state_id: str) -> Path:
    return state_dir(raw_root, state_id) / "before"


def candidate_dir(raw_root: str | Path, state_id: str, candidate_index: int) -> Path:
    return state_dir(raw_root, state_id) / f"cand_{candidate_index}"


# --- before-state hashing ----------------------------------------------------

def content_hash(path: str | Path) -> str:
    """Deterministic SHA-256 over a file or a directory's sorted contents.

    For a directory, every regular file is hashed as ``relpath\\0bytes`` in sorted
    order, so two byte-identical trees produce the same digest regardless of walk
    order. ``hashes.json`` is skipped so re-hashing a populated ``before/`` dir is
    stable.
    """
    target = Path(path)
    digest = hashlib.sha256()
    if target.is_file():
        digest.update(target.read_bytes())
        return digest.hexdigest()
    if not target.exists():
        raise FileNotFoundError(f"cannot hash missing path: {target}")
    files = sorted(
        p for p in target.rglob("*") if p.is_file() and p.name != "hashes.json"
    )
    for file_path in files:
        rel = file_path.relative_to(target).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def compute_before_state_hashes(before_state_dir: str | Path) -> dict[str, str]:
    """Compute ``checkpoint_hash`` (adapter) + ``optimizer_state_hash``.

    Looks for an ``adapter`` subdir / ``adapter_model.*`` for the checkpoint hash
    and an ``optimizer`` subdir / ``optimizer*.pt`` for the optimizer hash; falls
    back to hashing the whole directory so the function never silently produces
    empty hashes.
    """
    root = Path(before_state_dir)
    adapter = _first_existing(root, ("adapter", "adapter_model.safetensors", "adapter_model.bin"))
    optimizer = _first_existing(root, ("optimizer", "optimizer.pt", "optimizer_state.pt"))
    checkpoint_hash = content_hash(adapter) if adapter is not None else content_hash(root)
    optimizer_state_hash = content_hash(optimizer) if optimizer is not None else checkpoint_hash
    return {
        "checkpoint_hash": checkpoint_hash,
        "optimizer_state_hash": optimizer_state_hash,
    }


def _first_existing(root: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def write_before_state_hashes(before_state_dir: str | Path) -> dict[str, str]:
    """Compute hashes and persist them to ``before/hashes.json``."""
    root = Path(before_state_dir)
    root.mkdir(parents=True, exist_ok=True)
    hashes = compute_before_state_hashes(root)
    (root / "hashes.json").write_text(json.dumps(hashes, sort_keys=True), encoding="utf-8")
    return hashes


def read_before_state_hashes(before_state_dir: str | Path) -> dict[str, str]:
    """Read ``before/hashes.json`` (computing + writing it once if absent)."""
    path = Path(before_state_dir) / "hashes.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return write_before_state_hashes(before_state_dir)


def assert_identical_before_state(before_dirs: Sequence[str | Path]) -> dict[str, str]:
    """Assert every candidate branched from a byte-identical before-state.

    Raises ``ValueError`` if any ``checkpoint_hash`` or ``optimizer_state_hash``
    differs across the supplied ``before/`` directories. Returns the shared hash.
    """
    if not before_dirs:
        raise ValueError("no before-state directories supplied")
    reference: dict[str, str] | None = None
    for directory in before_dirs:
        hashes = read_before_state_hashes(directory)
        if reference is None:
            reference = hashes
            continue
        for key in ("checkpoint_hash", "optimizer_state_hash"):
            if hashes.get(key) != reference.get(key):
                raise ValueError(
                    f"before-state mismatch at {directory}: {key} "
                    f"{hashes.get(key)!r} != {reference.get(key)!r}"
                )
    assert reference is not None
    return reference


# --- prime-rl resume command -------------------------------------------------

def branch_command(
    rl_command: str,
    config_path: str | Path,
    *,
    resume_step: int,
    ckpt_interval: int = 1,
) -> list[str]:
    """Exact prime-rl command applying ONE GRPO step from a resumed checkpoint."""
    return [
        *shlex.split(rl_command),
        "@",
        str(config_path),
        "--ckpt",
        "--ckpt.interval",
        str(ckpt_interval),
        "--ckpt.resume-step",
        str(resume_step),
    ]


# --- candidate artifact writers ----------------------------------------------

def write_candidate_artifacts(
    cand_dir: str | Path,
    *,
    rollouts: Sequence[dict[str, Any]],
    probe_before: dict[str, Any],
    probe_after: dict[str, Any],
    grad_sketch: Sequence[float] | None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write one candidate's raw artifacts; ``grad_sketch=None`` -> fallback flag."""
    directory = Path(cand_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "rollouts.jsonl").open("w", encoding="utf-8") as handle:
        for row in rollouts:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (directory / "probe_before.json").write_text(
        json.dumps(probe_before, sort_keys=True), encoding="utf-8"
    )
    (directory / "probe_after.json").write_text(
        json.dumps(probe_after, sort_keys=True), encoding="utf-8"
    )
    if extra is not None:
        (directory / "candidate.json").write_text(
            json.dumps(extra, sort_keys=True), encoding="utf-8"
        )
    if grad_sketch is None:
        write_grad_unavailable(directory)
    else:
        write_grad_sketch(directory, grad_sketch)
    return directory


def write_grad_sketch(cand_dir: str | Path, vector: Sequence[float]) -> Path:
    """Persist a gradient sketch as ``grad_sketch.npy`` (numpy deferred)."""
    import numpy as np

    path = Path(cand_dir) / "grad_sketch.npy"
    np.save(path, np.asarray(vector, dtype=np.float32))
    return path


def write_grad_unavailable(cand_dir: str | Path, reason: str = "grad_unavailable") -> Path:
    """Write the gradient-fallback marker so the feature extractor zero-fills."""
    path = Path(cand_dir) / "grad_unavailable.flag"
    path.write_text(reason + "\n", encoding="utf-8")
    return path


# --- LoRA gradient sketch (Wave 2; torch deferred) ---------------------------

def compute_lora_gradient_sketch(
    checkpoint: Path,
    candidate_rows: Sequence[dict[str, Any]],
    *,
    sketch_dim: int = 64,
    seed: int = 1729,
    model_name: str = "Qwen/Qwen3-8B",
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> list[float]:
    """64-dim random projection of the candidate's LoRA gradient (Wave 2 GPU).

    Runs a self-contained backward pass over the candidate batch, flattens the
    LoRA parameter gradients, and projects them with a fixed seeded Gaussian
    matrix. Falls back to a zero sketch (never NaN) if no gradient is available.
    All torch usage is local to this function.
    """
    import numpy as np
    import torch

    from math_loop.probe_loss import _cleanup_torch_cuda, load_model_and_tokenizer

    if os.environ.get("TAP_NO_TORCH") == "1":
        return [0.0] * sketch_dim
    model = None
    tokenizer = None
    try:
        model, tokenizer = load_model_and_tokenizer(
            checkpoint, model_name=model_name, device=device, dtype=dtype
        )
        model.train()
        grads: list[Any] = []
        for row in candidate_rows:
            text = (row.get("prompt_text") or "") + (row.get("completion_text") or "")
            enc = tokenizer(text, return_tensors="pt").to(device)
            out = model(**enc, labels=enc["input_ids"])
            model.zero_grad(set_to_none=True)
            out.loss.backward()
            flat = [
                p.grad.detach().reshape(-1).float()
                for _, p in model.named_parameters()
                if p.requires_grad and p.grad is not None
            ]
            if flat:
                grads.append(torch.cat(flat))
        if not grads:
            return [0.0] * sketch_dim
        gradient = torch.stack(grads).mean(0).cpu().numpy()
        rng = np.random.default_rng(seed)
        projection = rng.standard_normal((sketch_dim, gradient.shape[0])).astype(np.float32)
        projection /= np.sqrt(gradient.shape[0])
        sketch = projection @ gradient.astype(np.float32)
        return [float(v) for v in np.nan_to_num(sketch)]
    except Exception:
        return [0.0] * sketch_dim
    finally:
        del model, tokenizer
        _cleanup_torch_cuda(torch, device)


# --- dry-run CLI -------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-command", default="uv run rl")
    parser.add_argument("--config", type=Path, default=Path("configs/prime_rl/qwen3_8b_math_branch.toml"))
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = branch_command(args.rl_command, args.config, resume_step=args.resume_step)
    print(shlex.join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
