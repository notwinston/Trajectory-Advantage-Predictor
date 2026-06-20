"""Synthetic TAP v1 data generator.

Emits the four schema-valid Parquet files (states/trajectories/candidates/
history) with correct dtypes/shapes and a plausible *learnable* signal, so the
parallel engine (W1a) and model (W1b) waves can build before real GPU
collection exists.

Self-consistency guarantees (checked by ``tests.test_tap_schema``):

* Joining ``candidates`` -> ``states`` on ``state_id``::

      utility_points == 1000 * (
          0.75 * (matched_probe_nll_before - matched_probe_nll_after)
        + 0.25 * (global_probe_nll_before  - global_probe_nll_after)
        - 0.05 * max(generic_kl_after - generic_kl_before, 0)
      )

  to within 1e-6.
* Candidate aggregates (reward/advantage/log-prob/entropy/sequence-length) equal
  the means over that candidate's 8 trajectories.
* Latent gains are a noisy linear function of
  ``[reward_mean, advantage_mean, -mean_log_prob, gradient_alignment,
  -semantic_similarity]`` plus a per-state intercept; the noise scale is tuned
  ONCE so a reference ridge's within-state Spearman at ``--seed 1729`` lands in
  ``[0.4, 0.8]`` (the promise does not gate on the exact value).

Numpy is imported at module level (pure CPU). No torch is needed here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

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

# Canonical (states_per_chain, candidates_per_state) for the three blessed label
# counts when chains == 2 (matches the spec's compressed scaling).
CANONICAL_LAYOUT = {48: (4, 6), 72: (6, 6), 128: (8, 8)}

SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
DIFFICULTIES = ("Level 3", "Level 4", "Level 5")

TRAJECTORIES_PER_CANDIDATE = 8  # 2 prompts x 4 completions
PROMPTS_PER_CANDIDATE = 2
HISTORY_WINDOW = 8  # latest eight applied update batches

# Latent linear weights on [reward_mean, advantage_mean, -mean_log_prob,
# gradient_alignment, -semantic_similarity]. Centered offset keeps utility near
# zero; GAIN_SCALE/NOISE_SCALE were tuned ONCE (ridge within-state Spearman in
# band at seed 1729).
_LATENT_WEIGHTS = np.array([0.9, 0.7, 0.5, 0.6, 0.4], dtype=np.float64)
_FEATURE_MEANS = np.array([0.5, 0.0, 0.7, 0.0, -0.4], dtype=np.float64)
GAIN_SCALE = 0.13
NOISE_SCALE = 0.030
STATE_INTERCEPT_SCALE = 0.10


def resolve_layout(labels: int, chains: int, candidates_per_state: int) -> tuple[int, int]:
    """Return ``(states_per_chain, candidates_per_state)`` enforcing divisibility.

    Tries the requested ``candidates_per_state`` first; if that does not divide
    evenly, falls back to the canonical layout for one of the blessed label
    counts (48/72/128) when ``chains == 2``.
    """
    if labels <= 0 or chains <= 0 or candidates_per_state <= 0:
        raise ValueError("labels, chains, candidates_per_state must be positive")
    if labels % chains != 0:
        raise ValueError(f"labels ({labels}) not divisible by chains ({chains})")
    per_chain = labels // chains
    if per_chain % candidates_per_state == 0:
        return per_chain // candidates_per_state, candidates_per_state
    if chains == 2 and labels in CANONICAL_LAYOUT:
        states, cps = CANONICAL_LAYOUT[labels]
        if states * cps * chains == labels:
            return states, cps
    raise ValueError(
        f"labels ({labels}) not divisible by chains*candidates_per_state "
        f"({chains}*{candidates_per_state}) and no canonical layout applies"
    )


def _percentiles(values: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.percentile(values, 10)),
        float(np.percentile(values, 50)),
        float(np.percentile(values, 90)),
    )


def _hash_like(rng: np.random.Generator) -> str:
    return "".join(rng.choice(list("0123456789abcdef"), size=16))


def generate(
    out_dir: str | Path,
    labels: int = 72,
    chains: int = 2,
    candidates_per_state: int = 6,
    seed: int = 1729,
) -> Dict[str, int]:
    """Generate the four Parquet files into ``out_dir``.

    Returns a dict of row counts per file.
    """
    states_per_chain, cps = resolve_layout(labels, chains, candidates_per_state)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Fixed low-rank directions correlating embeddings to utility.
    cand_dir = rng.standard_normal(CANDIDATE_EMBEDDING_DIM).astype(np.float64)
    cand_dir /= np.linalg.norm(cand_dir)
    grad_dir = rng.standard_normal(GRADIENT_SKETCH_DIM).astype(np.float64)
    grad_dir /= np.linalg.norm(grad_dir)
    traj_dir = rng.standard_normal(TRAJECTORY_EMBEDDING_DIM).astype(np.float64)
    traj_dir /= np.linalg.norm(traj_dir)

    states_rows: List[dict] = []
    trajectories_rows: List[dict] = []
    candidates_rows: List[dict] = []
    history_rows: List[dict] = []

    for chain_index in range(chains):
        chain_id = str(chain_index)
        applied: List[dict] = []  # records of applied candidates, most-recent last
        for state_index in range(states_per_chain):
            state_id = f"{chain_index}-{state_index}"
            step = state_index

            matched_before = float(rng.normal(1.0, 0.08))
            global_before = float(rng.normal(1.0, 0.08))
            generic_kl_before = float(abs(rng.normal(0.004, 0.002)))
            a_state = float(rng.normal(0.0, STATE_INTERCEPT_SCALE))

            history_for_state = applied[-HISTORY_WINDOW:][::-1]  # most recent first
            history_ids = [rec["candidate_id"] for rec in history_for_state]

            states_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "state_id": state_id,
                    "chain_id": chain_id,
                    "step": int(step),
                    "seed": int(seed + chain_index * 1000 + state_index),
                    "checkpoint_hash": _hash_like(rng),
                    "optimizer_state_hash": _hash_like(rng),
                    "learning_rate": float(1e-6 * rng.uniform(0.8, 1.2)),
                    "grpo_beta": 0.04,
                    "clip_range": 0.2,
                    "lora_rank": 16,
                    "matched_probe_nll_before": matched_before,
                    "global_probe_nll_before": global_before,
                    "generic_kl_before": generic_kl_before,
                    "adam_first_moment_norm": float(abs(rng.normal(0.5, 0.1))),
                    "adam_second_moment_norm": float(abs(rng.normal(0.3, 0.05))),
                    "policy_fingerprint": rng.standard_normal(POLICY_FINGERPRINT_DIM)
                    .astype(np.float32)
                    .tolist(),
                    "history_candidate_ids": history_ids,
                }
            )

            # --- generate this state's candidate branches ---
            state_candidates: List[dict] = []
            for k in range(cps):
                candidate_id = f"{state_id}-{k}"

                # Latent standard-normal factors -> displayed (linear) features.
                z = rng.standard_normal(5)
                target_reward = 0.5 + 0.12 * z[0]
                target_advantage = 0.25 * z[1]
                target_log_prob = -0.7 + 0.12 * z[2]
                grad_align = float(np.clip(0.15 * z[3], -1.0, 1.0))
                sem_sim = float(np.clip(0.4 + 0.12 * z[4], 0.0, 1.0))

                # --- 8 trajectories consistent with candidate-level targets ---
                traj_records = []
                adv_noise = rng.normal(0.0, 0.15, size=TRAJECTORIES_PER_CANDIDATE)
                adv_noise -= adv_noise.mean()  # group-centered advantages
                for j in range(TRAJECTORIES_PER_CANDIDATE):
                    prompt_index = j // (TRAJECTORIES_PER_CANDIDATE // PROMPTS_PER_CANDIDATE)
                    reward_total = float(np.clip(target_reward + rng.normal(0, 0.08), 0.0, 1.0))
                    reward_exact = float(1.0 if rng.uniform() < reward_total else 0.0)
                    reward_format = float(1.0 if rng.uniform() < 0.95 else 0.0)
                    advantage = float(target_advantage + adv_noise[j])
                    seq_len = int(rng.integers(24, 192))
                    mean_log_prob = float(target_log_prob + rng.normal(0, 0.05))
                    geo_mean_prob = float(np.exp(mean_log_prob))
                    arith_mean_prob = float(np.clip(geo_mean_prob * rng.uniform(1.0, 1.08), 0.0, 1.0))
                    token_lp = rng.normal(mean_log_prob, 0.2, size=max(seq_len, 4))
                    lp10, lp50, lp90 = _percentiles(token_lp)
                    mean_entropy = float(abs(rng.normal(1.0, 0.2)))
                    token_ent = abs(rng.normal(mean_entropy, 0.2, size=max(seq_len, 4)))
                    e10, e50, e90 = _percentiles(token_ent)
                    early_lp = float(mean_log_prob + rng.normal(0, 0.03))
                    late_lp = float(mean_log_prob + rng.normal(0, 0.03))
                    traj_signal = reward_total - 0.5
                    traj_emb = (
                        rng.standard_normal(TRAJECTORY_EMBEDDING_DIM) * 0.1
                        + traj_signal * 0.3 * traj_dir
                    ).astype(np.float32)
                    trajectory_id = f"{candidate_id}-t{j}"
                    record = {
                        "state_id": state_id,
                        "candidate_id": candidate_id,
                        "trajectory_id": trajectory_id,
                        "prompt_id": f"{candidate_id}-p{prompt_index}",
                        "subject": str(rng.choice(SUBJECTS)),
                        "difficulty": str(rng.choice(DIFFICULTIES)),
                        "prompt_text": f"[synthetic MATH prompt {candidate_id} p{prompt_index}]",
                        "completion_text": f"[synthetic completion {trajectory_id}] \\boxed{{{j}}}",
                        "reward_total": reward_total,
                        "reward_exact_answer": reward_exact,
                        "reward_format": reward_format,
                        "advantage": advantage,
                        "sequence_length": seq_len,
                        "mean_token_log_probability": mean_log_prob,
                        "geometric_mean_probability": geo_mean_prob,
                        "arithmetic_mean_probability": arith_mean_prob,
                        "token_log_probability_p10": lp10,
                        "token_log_probability_p50": lp50,
                        "token_log_probability_p90": lp90,
                        "mean_token_entropy": mean_entropy,
                        "entropy_p10": e10,
                        "entropy_p50": e50,
                        "entropy_p90": e90,
                        "early_mean_log_probability": early_lp,
                        "late_mean_log_probability": late_lp,
                        "confidence_slope": float(late_lp - early_lp),
                        "mean_old_to_current_log_ratio": float(rng.normal(0, 0.02)),
                        "mean_current_to_reference_log_ratio": float(rng.normal(0, 0.05)),
                        "clipped_token_fraction": float(rng.uniform(0, 0.1)),
                        "trajectory_embedding": traj_emb.tolist(),
                    }
                    traj_records.append(record)

                traj_records_np = {
                    key: np.array([r[key] for r in traj_records])
                    for key in (
                        "reward_total",
                        "advantage",
                        "mean_token_log_probability",
                        "arithmetic_mean_probability",
                        "mean_token_entropy",
                        "sequence_length",
                    )
                }
                reward_mean = float(traj_records_np["reward_total"].mean())
                reward_std = float(traj_records_np["reward_total"].std())
                advantage_mean = float(traj_records_np["advantage"].mean())
                advantage_std = float(traj_records_np["advantage"].std())
                mean_log_probability = float(traj_records_np["mean_token_log_probability"].mean())
                arithmetic_mean_probability = float(
                    traj_records_np["arithmetic_mean_probability"].mean()
                )
                mean_entropy = float(traj_records_np["mean_token_entropy"].mean())
                mean_sequence_length = float(traj_records_np["sequence_length"].mean())
                geometric_mean_probability = float(np.exp(mean_log_probability))

                # --- latent gains: noisy linear fn of the 5 features + intercept ---
                features = np.array(
                    [
                        reward_mean,
                        advantage_mean,
                        -mean_log_probability,
                        grad_align,
                        -sem_sim,
                    ],
                    dtype=np.float64,
                )
                signal = float((features - _FEATURE_MEANS) @ _LATENT_WEIGHTS)
                matched_gain = (
                    GAIN_SCALE * (signal + a_state) + NOISE_SCALE * float(rng.standard_normal())
                )
                global_gain = 0.5 * matched_gain + 0.01 * float(rng.standard_normal())
                incremental_generic_kl = float(abs(rng.normal(0.002, 0.0015)))

                matched_after = matched_before - matched_gain
                global_after = global_before - global_gain
                generic_kl_after = generic_kl_before + incremental_generic_kl
                utility_points = 1000.0 * (
                    0.75 * matched_gain
                    + 0.25 * global_gain
                    - 0.05 * max(incremental_generic_kl, 0.0)
                )

                gradient_norm = float(abs(rng.normal(1.0, 0.2)))
                estimated_update_norm = float(abs(rng.normal(0.5, 0.1)))
                mean_grad_sim = float(np.clip(grad_align + rng.normal(0, 0.05), -1.0, 1.0))
                mean_sem_sim = float(np.clip(sem_sim - abs(rng.normal(0, 0.05)), 0.0, 1.0))

                cand_emb = (
                    rng.standard_normal(CANDIDATE_EMBEDDING_DIM) * 0.1
                    + (matched_gain * 5.0) * cand_dir
                ).astype(np.float32)
                grad_sketch = (
                    rng.standard_normal(GRADIENT_SKETCH_DIM) * 0.1 + grad_align * grad_dir
                ).astype(np.float32)
                candidate_log_probability_change = float(rng.normal(0.0, 0.05))

                candidate_record = {
                    "state_id": state_id,
                    "candidate_id": candidate_id,
                    "chain_id": chain_id,
                    "step": int(step),
                    "trajectory_ids": [r["trajectory_id"] for r in traj_records],
                    "candidate_reward_mean": reward_mean,
                    "candidate_reward_std": reward_std,
                    "candidate_advantage_mean": advantage_mean,
                    "candidate_advantage_std": advantage_std,
                    "candidate_mean_log_probability": mean_log_probability,
                    "candidate_geometric_mean_probability": geometric_mean_probability,
                    "candidate_arithmetic_mean_probability": arithmetic_mean_probability,
                    "candidate_mean_entropy": mean_entropy,
                    "candidate_mean_sequence_length": mean_sequence_length,
                    "candidate_embedding": cand_emb.tolist(),
                    "gradient_sketch": grad_sketch.tolist(),
                    "gradient_norm": gradient_norm,
                    "estimated_update_norm": estimated_update_norm,
                    "max_semantic_similarity_to_history": sem_sim,
                    "mean_semantic_similarity_to_history": mean_sem_sim,
                    "max_gradient_similarity_to_history": grad_align,
                    "mean_gradient_similarity_to_history": mean_grad_sim,
                    "matched_probe_nll_after": matched_after,
                    "global_probe_nll_after": global_after,
                    "generic_kl_after": generic_kl_after,
                    "matched_gain": matched_gain,
                    "global_gain": global_gain,
                    "incremental_generic_kl": incremental_generic_kl,
                    "utility_points": utility_points,
                    "candidate_log_probability_change": candidate_log_probability_change,
                    "matched_exact_match_before": float(rng.integers(0, 9)) / 8.0,
                    "matched_exact_match_after": float(rng.integers(0, 9)) / 8.0,
                    "is_selected_for_main_chain": False,
                }
                state_candidates.append(candidate_record)
                trajectories_rows.extend(traj_records)
                candidates_rows.append(candidate_record)

            # --- choose exactly one candidate to apply to the main chain ---
            chosen = int(rng.integers(0, cps))
            for idx, cand in enumerate(state_candidates):
                cand["is_selected_for_main_chain"] = idx == chosen
            chosen_cand = state_candidates[chosen]
            applied.append(
                {
                    "candidate_id": chosen_cand["candidate_id"],
                    "candidate_embedding": chosen_cand["candidate_embedding"],
                    "gradient_sketch": chosen_cand["gradient_sketch"],
                    "reward_mean": chosen_cand["candidate_reward_mean"],
                    "advantage_mean": chosen_cand["candidate_advantage_mean"],
                    "mean_log_probability": chosen_cand["candidate_mean_log_probability"],
                    "mean_entropy": chosen_cand["candidate_mean_entropy"],
                    "update_norm": chosen_cand["estimated_update_norm"],
                    "loss_change": -chosen_cand["matched_gain"],
                    "log_probability_change": chosen_cand["candidate_log_probability_change"],
                }
            )

            # --- history rows for THIS state (applied updates before this state) ---
            for position, rec in enumerate(history_for_state):
                history_rows.append(
                    {
                        "state_id": state_id,
                        "history_position": int(position),
                        "relative_age": int(position + 1),
                        "historical_candidate_id": rec["candidate_id"],
                        "historical_candidate_embedding": rec["candidate_embedding"],
                        "historical_gradient_sketch": rec["gradient_sketch"],
                        "historical_reward_mean": rec["reward_mean"],
                        "historical_advantage_mean": rec["advantage_mean"],
                        "historical_mean_log_probability": rec["mean_log_probability"],
                        "historical_mean_entropy": rec["mean_entropy"],
                        "historical_update_norm": rec["update_norm"],
                        "historical_training_loss_change": rec["loss_change"],
                        "historical_candidate_log_probability_change": rec["log_probability_change"],
                    }
                )

    _write_parquet(out_path / "states.parquet", "states.parquet", states_rows)
    _write_parquet(out_path / "trajectories.parquet", "trajectories.parquet", trajectories_rows)
    _write_parquet(out_path / "candidates.parquet", "candidates.parquet", candidates_rows)
    _write_parquet(out_path / "history.parquet", "history.parquet", history_rows)

    return {
        "states.parquet": len(states_rows),
        "trajectories.parquet": len(trajectories_rows),
        "candidates.parquet": len(candidates_rows),
        "history.parquet": len(history_rows),
    }


def _write_parquet(path: Path, file_name: str, rows: List[dict]) -> None:
    schema = arrow_schema(file_name)
    columns = {name: [row[name] for row in rows] for name in schema.names}
    table = pa.Table.from_pydict(columns, schema=schema)
    pq.write_table(table, path)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic TAP v1 Parquet data.")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--labels", type=int, default=192, help="total candidate labels")
    parser.add_argument("--chains", type=int, default=3)
    parser.add_argument("--candidates-per-state", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)

    counts = generate(
        out_dir=args.out,
        labels=args.labels,
        chains=args.chains,
        candidates_per_state=args.candidates_per_state,
        seed=args.seed,
    )
    print(
        f"Wrote {args.out}: "
        + ", ".join(f"{name}={count}" for name, count in counts.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
