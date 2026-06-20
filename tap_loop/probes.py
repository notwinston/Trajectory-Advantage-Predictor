"""Probe selection and utility-label math for TAP v1."""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

import numpy as np


def _row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("subject", "")), int(row.get("level", row.get("difficulty", 0)))


def _dedupe(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row.get("id"))
        if row_id not in seen:
            seen.add(row_id)
            output.append(row)
    return output


def select_global_probe(heldout_rows: Sequence[dict[str, Any]], *, size: int = 8, seed: int = 1729) -> list[dict[str, Any]]:
    if size <= 0:
        raise ValueError("global probe size must be positive")
    rows = list(heldout_rows)
    if len(rows) < size:
        raise ValueError(f"need at least {size} heldout rows, got {len(rows)}")
    rng = random.Random(seed)
    buckets: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(int(row["level"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for level in sorted(buckets):
        bucket = list(buckets[level])
        rng.shuffle(bucket)
        if bucket:
            selected.append(bucket[0])
    remaining = [row for row in rows if row not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, size - len(selected))])
    return _dedupe(selected)[:size]


def select_matched_probe(
    candidate_prompts: Sequence[dict[str, Any]],
    heldout_rows: Sequence[dict[str, Any]],
    *,
    size: int = 8,
    seed: int = 1729,
) -> list[dict[str, Any]]:
    if not candidate_prompts:
        raise ValueError("candidate prompts are required for matched probe selection")
    if len(heldout_rows) < size:
        raise ValueError(f"need at least {size} heldout rows, got {len(heldout_rows)}")

    rng = random.Random(seed)
    subjects = sorted({str(row["subject"]) for row in candidate_prompts})
    levels = sorted({int(row["level"]) for row in candidate_prompts})
    subject_quota = {subject: size // len(subjects) for subject in subjects}
    for subject in subjects[: size % len(subjects)]:
        subject_quota[subject] += 1

    selected: list[dict[str, Any]] = []
    for subject, quota in subject_quota.items():
        same_subject = [row for row in heldout_rows if str(row.get("subject")) == subject]
        same_level = [row for row in same_subject if int(row.get("level", 0)) in levels]
        rng.shuffle(same_level)
        rng.shuffle(same_subject)
        selected.extend(same_level[:quota])
        subject_count = len([row for row in _dedupe(selected) if str(row.get("subject")) == subject])
        if subject_count < quota:
            selected.extend(same_subject[: quota - subject_count])

    selected = _dedupe(selected)
    if len(selected) < size:
        fallback = list(heldout_rows)
        rng.shuffle(fallback)
        selected.extend(row for row in fallback if row not in selected)
    return _dedupe(selected)[:size]


def utility_points(
    matched_nll_before: float,
    matched_nll_after: float,
    global_nll_before: float,
    global_nll_after: float,
    generic_kl_before: float,
    generic_kl_after: float,
) -> dict[str, float]:
    matched_gain = matched_nll_before - matched_nll_after
    global_gain = global_nll_before - global_nll_after
    incremental_generic_kl = generic_kl_after - generic_kl_before
    utility = 1000.0 * (
        0.8 * matched_gain + 0.2 * global_gain - 0.03 * max(incremental_generic_kl, 0.0)
    )
    return {
        "matched_gain": matched_gain,
        "global_gain": global_gain,
        "incremental_generic_kl": incremental_generic_kl,
        "utility_points": utility,
    }


def average_token_kl(reference_log_probs: Sequence[Sequence[float]], policy_log_probs: Sequence[Sequence[float]]) -> float:
    """Compute average KL(reference || policy) from per-token log-probability rows."""

    ref = np.asarray(reference_log_probs, dtype=np.float64)
    pol = np.asarray(policy_log_probs, dtype=np.float64)
    if ref.shape != pol.shape:
        raise ValueError(f"KL tensors must have the same shape, got {ref.shape} and {pol.shape}")
    if ref.ndim != 2:
        raise ValueError("KL tensors must be rank-2: tokens x vocab/options")
    ref_probs = np.exp(ref - ref.max(axis=1, keepdims=True))
    ref_probs /= ref_probs.sum(axis=1, keepdims=True)
    kl = ref_probs * (ref - pol)
    return float(np.mean(np.sum(kl, axis=1)))


def geometric_mean_probability(mean_token_log_probability: float) -> float:
    return float(math.exp(mean_token_log_probability))
