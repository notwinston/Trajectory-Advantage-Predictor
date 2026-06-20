"""TAP v1 feature extractor: raw branch artifacts -> 4 schema-valid Parquet files.

Reads the tree emitted by :mod:`math_loop.branch` /
:mod:`math_loop.tap_controller`::

    <raw_root>/state_<chain>-<state>/
        before/hashes.json
        state.json
        cand_<k>/{rollouts.jsonl, probe_before.json, probe_after.json,
                  grad_sketch.npy | grad_unavailable.flag, candidate.json?}

and writes ``states/trajectories/candidates/history.parquet`` conforming to the
frozen contract in :mod:`tap.schema`.

Every "fancy" feature is flag-gated with a documented, NaN-free fallback so the
Parquet is ALWAYS schema-valid:

* ``gradient_sketch`` — read ``grad_sketch.npy``; if ``grad_unavailable.flag`` is
  present or ``TAP_NO_TORCH=1``, emit a zeroed (flagged) 64-vector.
* ``trajectory_embedding`` / ``candidate_embedding`` — use values carried in the
  artifacts when present, else derive deterministically (seeded, finite).
* token-level stats fall back to carried scalars, then to 0.0.

numpy/pyarrow are imported at module level (this runs in the CPU venv). torch is
never imported here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tap.schema import (
    CANDIDATE_EMBEDDING_DIM,
    GRADIENT_SKETCH_DIM,
    POLICY_FINGERPRINT_DIM,
    SCHEMA_VERSION,
    TRAJECTORY_EMBEDDING_DIM,
    arrow_schema,
)

UTILITY_SCALE = 1000.0
MATCHED_WEIGHT = 0.75
GLOBAL_WEIGHT = 0.25
KL_PENALTY = 0.05

# Fixed projection seed so candidate_embedding (256) derivation is reproducible.
_PROJECTION_SEED = 20240601


# --- deterministic fallbacks -------------------------------------------------

def _seed_from_key(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def _finite(values: Sequence[float]) -> list[float]:
    arr = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return [float(v) for v in arr]


def _fixed_width(values: Sequence[float], width: int) -> list[float]:
    """Coerce ``values`` to exactly ``width`` finite float32 entries."""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape[0] < width:
        arr = np.concatenate([arr, np.zeros(width - arr.shape[0], dtype=np.float32)])
    else:
        arr = arr[:width]
    return _finite(arr)


def _derive_vector(key: str, dim: int, signal: float = 0.0) -> list[float]:
    rng = np.random.default_rng(_seed_from_key(key))
    vec = rng.standard_normal(dim).astype(np.float32) * 0.1 + np.float32(signal)
    return _finite(vec)


def _project_to(vec: Sequence[float], out_dim: int) -> list[float]:
    source = np.asarray(vec, dtype=np.float32).reshape(-1)
    if source.shape[0] == 0:
        return [0.0] * out_dim
    rng = np.random.default_rng(_PROJECTION_SEED)
    projection = rng.standard_normal((out_dim, source.shape[0])).astype(np.float32)
    projection /= math.sqrt(source.shape[0])
    return _finite(projection @ source)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(av)
    nb = np.linalg.norm(bv)
    if na == 0.0 or nb == 0.0:
        return 0.0
    value = float(np.dot(av, bv) / (na * nb))
    if not math.isfinite(value):
        return 0.0
    return value


def _percentiles(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return (
        float(np.percentile(arr, 10)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 90)),
    )


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    if len(values) == 0:
        return default
    return float(np.mean(np.asarray(values, dtype=np.float64)))


# --- IO ----------------------------------------------------------------------

def _read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _state_sort_key(state_dir: Path) -> tuple[int, int, str]:
    name = state_dir.name[len("state_") :]
    try:
        chain_str, state_str = name.split("-", 1)
        return int(chain_str), int(state_str), name
    except ValueError:
        return 0, 0, name


# --- utility -----------------------------------------------------------------

def compute_utility_points(
    matched_before: float,
    matched_after: float,
    global_before: float,
    global_after: float,
    generic_before: float,
    generic_after: float,
) -> dict[str, float]:
    """Spec UTILITY TARGET, computed via the (states -> candidates) join values."""
    matched_gain = matched_before - matched_after
    global_gain = global_before - global_after
    incremental_generic_kl = generic_after - generic_before
    utility_points = UTILITY_SCALE * (
        MATCHED_WEIGHT * matched_gain
        + GLOBAL_WEIGHT * global_gain
        - KL_PENALTY * max(incremental_generic_kl, 0.0)
    )
    return {
        "matched_gain": matched_gain,
        "global_gain": global_gain,
        "incremental_generic_kl": incremental_generic_kl,
        "utility_points": utility_points,
    }


# --- trajectory features -----------------------------------------------------

def _trajectory_row(row: dict[str, Any], state_id: str, candidate_id: str) -> dict[str, Any]:
    token_lp = [float(v) for v in row.get("token_log_probabilities", [])]
    token_ent = [float(v) for v in row.get("token_entropies", [])]
    seq_len = int(row.get("sequence_length", len(token_lp) or 0))

    if token_lp:
        mean_lp = _mean(token_lp)
        arith_mean_prob = float(np.mean(np.exp(np.asarray(token_lp, dtype=np.float64))))
        half = max(len(token_lp) // 2, 1)
        early_lp = _mean(token_lp[:half])
        late_lp = _mean(token_lp[half:]) if len(token_lp) > half else early_lp
        lp10, lp50, lp90 = _percentiles(token_lp)
    else:
        mean_lp = float(row.get("mean_token_log_probability", 0.0))
        arith_mean_prob = float(row.get("arithmetic_mean_probability", math.exp(mean_lp)))
        early_lp = float(row.get("early_mean_log_probability", mean_lp))
        late_lp = float(row.get("late_mean_log_probability", mean_lp))
        lp10 = lp50 = lp90 = mean_lp

    if token_ent:
        mean_ent = _mean(token_ent)
        e10, e50, e90 = _percentiles(token_ent)
    else:
        mean_ent = float(row.get("mean_token_entropy", 0.0))
        e10 = e50 = e90 = mean_ent

    old_to_cur = row.get("old_to_current_log_ratios")
    cur_to_ref = row.get("current_to_reference_log_ratios")
    mean_old_to_cur = (
        _mean([float(v) for v in old_to_cur])
        if isinstance(old_to_cur, list)
        else float(row.get("mean_old_to_current_log_ratio", 0.0))
    )
    mean_cur_to_ref = (
        _mean([float(v) for v in cur_to_ref])
        if isinstance(cur_to_ref, list)
        else float(row.get("mean_current_to_reference_log_ratio", 0.0))
    )

    traj_id = str(row.get("trajectory_id", f"{candidate_id}-t?"))
    embedding = row.get("trajectory_embedding")
    if isinstance(embedding, list) and embedding:
        traj_emb = _fixed_width(embedding, TRAJECTORY_EMBEDDING_DIM)
    else:
        signal = float(row.get("reward_total", 0.0)) - 0.5
        traj_emb = _derive_vector(traj_id, TRAJECTORY_EMBEDDING_DIM, signal=0.3 * signal)

    return {
        "state_id": state_id,
        "candidate_id": candidate_id,
        "trajectory_id": traj_id,
        "prompt_id": str(row.get("prompt_id", f"{candidate_id}-p0")),
        "subject": str(row.get("subject", "unknown")),
        "difficulty": str(row.get("difficulty", "unknown")),
        "prompt_text": str(row.get("prompt_text", "")),
        "completion_text": str(row.get("completion_text", "")),
        "reward_total": float(row.get("reward_total", 0.0)),
        "reward_exact_answer": float(row.get("reward_exact_answer", 0.0)),
        "reward_format": float(row.get("reward_format", 0.0)),
        "advantage": float(row.get("advantage", 0.0)),
        "sequence_length": seq_len,
        "mean_token_log_probability": mean_lp,
        "geometric_mean_probability": float(math.exp(mean_lp)),
        "arithmetic_mean_probability": arith_mean_prob,
        "token_log_probability_p10": lp10,
        "token_log_probability_p50": lp50,
        "token_log_probability_p90": lp90,
        "mean_token_entropy": mean_ent,
        "entropy_p10": e10,
        "entropy_p50": e50,
        "entropy_p90": e90,
        "early_mean_log_probability": early_lp,
        "late_mean_log_probability": late_lp,
        "confidence_slope": float(late_lp - early_lp),
        "mean_old_to_current_log_ratio": mean_old_to_cur,
        "mean_current_to_reference_log_ratio": mean_cur_to_ref,
        "clipped_token_fraction": float(row.get("clipped_token_fraction", 0.0)),
        "trajectory_embedding": traj_emb,
    }


def _load_gradient_sketch(cand_dir: Path) -> list[float]:
    """Read grad_sketch.npy, or zero-fill on fallback flag / TAP_NO_TORCH."""
    if os.environ.get("TAP_NO_TORCH") == "1":
        return [0.0] * GRADIENT_SKETCH_DIM
    if (cand_dir / "grad_unavailable.flag").exists():
        return [0.0] * GRADIENT_SKETCH_DIM
    npy = cand_dir / "grad_sketch.npy"
    if not npy.exists():
        return [0.0] * GRADIENT_SKETCH_DIM
    try:
        arr = np.load(npy)
    except Exception:
        return [0.0] * GRADIENT_SKETCH_DIM
    return _fixed_width(arr, GRADIENT_SKETCH_DIM)


# --- conversion --------------------------------------------------------------

def convert(raw_root: str | Path, out_dir: str | Path) -> dict[str, int]:
    """Convert the raw-artifact tree under ``raw_root`` into the 4 Parquet files."""
    raw = Path(raw_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    state_dirs = sorted(
        (p for p in raw.glob("state_*") if p.is_dir()), key=_state_sort_key
    )

    states_rows: list[dict[str, Any]] = []
    trajectories_rows: list[dict[str, Any]] = []
    candidates_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []

    for state_path in state_dirs:
        state = _read_json(state_path / "state.json")
        state_id = str(state.get("state_id", state_path.name[len("state_") :]))
        chain_id = str(state.get("chain_id", state_id.split("-", 1)[0]))
        hashes = _read_json(state_path / "before" / "hashes.json")

        cand_dirs = sorted(
            (p for p in state_path.glob("cand_*") if p.is_dir()),
            key=lambda p: int(p.name[len("cand_") :]) if p.name[len("cand_") :].isdigit() else 0,
        )

        # before-probe values (state-level; identical across candidates).
        first_before = _read_json(cand_dirs[0] / "probe_before.json") if cand_dirs else {}
        matched_before = float(state.get("matched_probe_nll_before", first_before.get("matched_probe_nll", 0.0)))
        global_before = float(state.get("global_probe_nll_before", first_before.get("global_probe_nll", 0.0)))
        generic_before = float(state.get("generic_kl_before", first_before.get("generic_kl", 0.0)))

        # history records (last 8 applied updates), attached to this state.
        history = state.get("history", []) or []
        history_ids = [str(rec.get("historical_candidate_id", rec.get("candidate_id", ""))) for rec in history]

        fingerprint = state.get("policy_fingerprint")
        if isinstance(fingerprint, list) and fingerprint:
            fingerprint_vec = _fixed_width(fingerprint, POLICY_FINGERPRINT_DIM)
        else:
            fingerprint_vec = _derive_vector(f"fingerprint::{state_id}", POLICY_FINGERPRINT_DIM)

        states_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "state_id": state_id,
                "chain_id": chain_id,
                "step": int(state.get("step", 0)),
                "seed": int(state.get("seed", 0)),
                "checkpoint_hash": str(hashes.get("checkpoint_hash", "")),
                "optimizer_state_hash": str(hashes.get("optimizer_state_hash", "")),
                "learning_rate": float(state.get("learning_rate", 0.0)),
                "grpo_beta": float(state.get("grpo_beta", 0.0)),
                "clip_range": float(state.get("clip_range", 0.0)),
                "lora_rank": int(state.get("lora_rank", 16)),
                "matched_probe_nll_before": matched_before,
                "global_probe_nll_before": global_before,
                "generic_kl_before": generic_before,
                "adam_first_moment_norm": float(state.get("adam_first_moment_norm", 0.0)),
                "adam_second_moment_norm": float(state.get("adam_second_moment_norm", 0.0)),
                "policy_fingerprint": fingerprint_vec,
                "history_candidate_ids": history_ids,
            }
        )

        for position, rec in enumerate(history):
            hist_emb = rec.get("historical_candidate_embedding") or rec.get("candidate_embedding")
            hist_grad = rec.get("historical_gradient_sketch") or rec.get("gradient_sketch")
            history_rows.append(
                {
                    "state_id": state_id,
                    "history_position": int(rec.get("history_position", position)),
                    "relative_age": int(rec.get("relative_age", position + 1)),
                    "historical_candidate_id": str(rec.get("historical_candidate_id", rec.get("candidate_id", ""))),
                    "historical_candidate_embedding": _fixed_width(hist_emb, CANDIDATE_EMBEDDING_DIM)
                    if isinstance(hist_emb, list) and hist_emb
                    else [0.0] * CANDIDATE_EMBEDDING_DIM,
                    "historical_gradient_sketch": _fixed_width(hist_grad, GRADIENT_SKETCH_DIM)
                    if isinstance(hist_grad, list) and hist_grad
                    else [0.0] * GRADIENT_SKETCH_DIM,
                    "historical_reward_mean": float(rec.get("historical_reward_mean", rec.get("reward_mean", 0.0))),
                    "historical_advantage_mean": float(rec.get("historical_advantage_mean", rec.get("advantage_mean", 0.0))),
                    "historical_mean_log_probability": float(rec.get("historical_mean_log_probability", rec.get("mean_log_probability", 0.0))),
                    "historical_mean_entropy": float(rec.get("historical_mean_entropy", rec.get("mean_entropy", 0.0))),
                    "historical_update_norm": float(rec.get("historical_update_norm", rec.get("update_norm", 0.0))),
                    "historical_training_loss_change": float(rec.get("historical_training_loss_change", rec.get("loss_change", 0.0))),
                    "historical_candidate_log_probability_change": float(rec.get("historical_candidate_log_probability_change", rec.get("log_probability_change", 0.0))),
                }
            )

        selected_index = int(state.get("selected_candidate_index", -1))

        for cand_path in cand_dirs:
            candidate_index = int(cand_path.name[len("cand_") :]) if cand_path.name[len("cand_") :].isdigit() else 0
            candidate_id = f"{state_id}-{candidate_index}"
            extra = _read_json(cand_path / "candidate.json")
            probe_after = _read_json(cand_path / "probe_after.json")

            traj_raw = _read_jsonl(cand_path / "rollouts.jsonl")
            traj_features = [_trajectory_row(row, state_id, candidate_id) for row in traj_raw]
            trajectories_rows.extend(traj_features)
            trajectory_ids = [t["trajectory_id"] for t in traj_features]

            rewards = [t["reward_total"] for t in traj_features]
            advantages = [t["advantage"] for t in traj_features]
            traj_mean_lps = [t["mean_token_log_probability"] for t in traj_features]
            arith = [t["arithmetic_mean_probability"] for t in traj_features]
            entropies = [t["mean_token_entropy"] for t in traj_features]
            seq_lens = [t["sequence_length"] for t in traj_features]

            candidate_mean_log_probability = _mean(traj_mean_lps)
            candidate_reward_mean = _mean(rewards)
            candidate_advantage_mean = _mean(advantages)

            # candidate_embedding (256): pool trajectory embeddings -> project.
            if traj_features:
                pooled = np.mean(
                    np.asarray([t["trajectory_embedding"] for t in traj_features], dtype=np.float32),
                    axis=0,
                )
            else:
                pooled = np.zeros(TRAJECTORY_EMBEDDING_DIM, dtype=np.float32)
            stored_cand_emb = extra.get("candidate_embedding")
            if isinstance(stored_cand_emb, list) and stored_cand_emb:
                candidate_embedding = _fixed_width(stored_cand_emb, CANDIDATE_EMBEDDING_DIM)
            else:
                candidate_embedding = _project_to(pooled, CANDIDATE_EMBEDDING_DIM)

            gradient_sketch = _load_gradient_sketch(cand_path)
            grad_arr = np.asarray(gradient_sketch, dtype=np.float64)
            gradient_norm = float(extra.get("gradient_norm", float(np.linalg.norm(grad_arr))))
            learning_rate = float(state.get("learning_rate", 0.0))
            estimated_update_norm = float(
                extra.get("estimated_update_norm", learning_rate * gradient_norm)
            )

            # similarity to history (semantic = embeddings, gradient = sketches).
            sem_sims = [
                _cosine(candidate_embedding, h["historical_candidate_embedding"]) for h in history_rows
                if h["state_id"] == state_id
            ]
            grad_sims = [
                _cosine(gradient_sketch, h["historical_gradient_sketch"]) for h in history_rows
                if h["state_id"] == state_id
            ]
            max_sem = max(sem_sims) if sem_sims else 0.0
            mean_sem = _mean(sem_sims) if sem_sims else 0.0
            max_grad = max(grad_sims) if grad_sims else 0.0
            mean_grad = _mean(grad_sims) if grad_sims else 0.0

            matched_after = float(probe_after.get("matched_probe_nll", matched_before))
            global_after = float(probe_after.get("global_probe_nll", global_before))
            generic_after = float(probe_after.get("generic_kl", generic_before))
            util = compute_utility_points(
                matched_before, matched_after, global_before, global_after, generic_before, generic_after
            )

            matched_before_probe = _read_json(cand_path / "probe_before.json")

            candidates_rows.append(
                {
                    "state_id": state_id,
                    "candidate_id": candidate_id,
                    "chain_id": chain_id,
                    "step": int(state.get("step", 0)),
                    "trajectory_ids": trajectory_ids,
                    "candidate_reward_mean": candidate_reward_mean,
                    "candidate_reward_std": float(np.std(np.asarray(rewards, dtype=np.float64))) if rewards else 0.0,
                    "candidate_advantage_mean": candidate_advantage_mean,
                    "candidate_advantage_std": float(np.std(np.asarray(advantages, dtype=np.float64))) if advantages else 0.0,
                    "candidate_mean_log_probability": candidate_mean_log_probability,
                    "candidate_geometric_mean_probability": float(math.exp(candidate_mean_log_probability)),
                    "candidate_arithmetic_mean_probability": _mean(arith),
                    "candidate_mean_entropy": _mean(entropies),
                    "candidate_mean_sequence_length": _mean(seq_lens),
                    "candidate_embedding": candidate_embedding,
                    "gradient_sketch": gradient_sketch,
                    "gradient_norm": gradient_norm,
                    "estimated_update_norm": estimated_update_norm,
                    "max_semantic_similarity_to_history": max_sem,
                    "mean_semantic_similarity_to_history": mean_sem,
                    "max_gradient_similarity_to_history": max_grad,
                    "mean_gradient_similarity_to_history": mean_grad,
                    "matched_probe_nll_after": matched_after,
                    "global_probe_nll_after": global_after,
                    "generic_kl_after": generic_after,
                    "matched_gain": util["matched_gain"],
                    "global_gain": util["global_gain"],
                    "incremental_generic_kl": util["incremental_generic_kl"],
                    "utility_points": util["utility_points"],
                    "candidate_log_probability_change": float(extra.get("candidate_log_probability_change", 0.0)),
                    "matched_exact_match_before": float(matched_before_probe.get("matched_exact_match", 0.0)),
                    "matched_exact_match_after": float(probe_after.get("matched_exact_match", 0.0)),
                    "is_selected_for_main_chain": bool(candidate_index == selected_index),
                }
            )

    _write_parquet(out / "states.parquet", "states.parquet", states_rows)
    _write_parquet(out / "trajectories.parquet", "trajectories.parquet", trajectories_rows)
    _write_parquet(out / "candidates.parquet", "candidates.parquet", candidates_rows)
    _write_parquet(out / "history.parquet", "history.parquet", history_rows)

    return {
        "states.parquet": len(states_rows),
        "trajectories.parquet": len(trajectories_rows),
        "candidates.parquet": len(candidates_rows),
        "history.parquet": len(history_rows),
    }


def _write_parquet(path: Path, file_name: str, rows: list[dict[str, Any]]) -> None:
    schema = arrow_schema(file_name)
    columns = {name: [row[name] for row in rows] for name in schema.names}
    table = pa.Table.from_pydict(columns, schema=schema)
    pq.write_table(table, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TAP v1 raw-artifacts -> 4 Parquet.")
    parser.add_argument("--raw", required=True, help="raw-artifact root (contains state_* dirs)")
    parser.add_argument("--out", required=True, help="output directory for the 4 parquet files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    counts = convert(args.raw, args.out)
    print(
        f"Wrote {args.out}: " + ", ".join(f"{name}={count}" for name, count in counts.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
