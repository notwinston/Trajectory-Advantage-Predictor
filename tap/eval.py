"""Within-state ranking evaluation for TAP v1.

Consumes only a model's ``score()`` output (``{state_id: {candidate_id: float}}``)
plus a truth table of the same shape holding ``utility_points``. Per the spec,
ranking quality matters more than absolute R^2 with ~72 noisy labels, so the
metrics are all *within-state*:

* ``spearman``          : mean within-state Spearman rank correlation
* ``pair_acc``          : mean within-state pairwise ranking accuracy
* ``top1_regret``       : mean (best true utility - true utility of model pick)
* ``mean_true_utility`` : mean true utility of the model-selected candidate
* ``lift_random`` / ``lift_reward`` / ``lift_prob`` : ``mean_true_utility`` minus
  the same metric for the random / reward-only / probability-only selectors.

Aggregation averages the per-state metrics, then averages across the two chain
directions. Pure numpy — no scipy dependency.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

TruthTable = Dict[str, Dict[str, float]]
ScoreDict = Dict[str, Dict[str, float]]

METRIC_COLS = (
    "spearman",
    "pair_acc",
    "top1_regret",
    "mean_true_utility",
    "lift_random",
    "lift_reward",
    "lift_prob",
)


def build_truth(candidates_df: pd.DataFrame) -> TruthTable:
    """{state_id: {candidate_id: utility_points}} from a candidates frame."""
    truth: TruthTable = {}
    for state_id, candidate_id, util in zip(
        candidates_df["state_id"].to_numpy(),
        candidates_df["candidate_id"].to_numpy(),
        candidates_df["utility_points"].to_numpy(),
    ):
        truth.setdefault(str(state_id), {})[str(candidate_id)] = float(util)
    return truth


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks (1-based), matching scipy.stats.rankdata('average')."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman(pred: np.ndarray, true: np.ndarray) -> float:
    """Spearman rank correlation; 0.0 when either side is constant."""
    if len(pred) < 2:
        return 0.0
    rp = _rankdata(pred)
    rt = _rankdata(true)
    if rp.std() == 0 or rt.std() == 0:
        return 0.0
    return float(np.corrcoef(rp, rt)[0, 1])


def pairwise_accuracy(pred: np.ndarray, true: np.ndarray) -> float | None:
    """Fraction of comparable pairs where pred and true agree on order.

    Pairs with equal true utility are skipped; ties in pred count as 0.5.
    Returns ``None`` if there are no comparable pairs.
    """
    n = len(pred)
    agree = 0.0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            dt = true[i] - true[j]
            if dt == 0:
                continue
            dp = pred[i] - pred[j]
            total += 1
            if dp == 0:
                agree += 0.5
            elif (dp > 0) == (dt > 0):
                agree += 1.0
    if total == 0:
        return None
    return agree / total


def _selected_index(scores: np.ndarray) -> int:
    """Argmax with deterministic tie-break (lowest index)."""
    return int(np.argmax(scores))


def evaluate(score_dict: ScoreDict, truth: TruthTable) -> Dict[str, float]:
    """Per-direction metrics (no lift; lift is added by :func:`aggregate`)."""
    spearmans: List[float] = []
    pair_accs: List[float] = []
    regrets: List[float] = []
    selected_utils: List[float] = []

    for state_id, true_map in truth.items():
        score_map = score_dict.get(state_id, {})
        candidate_ids = sorted(true_map.keys())
        true = np.array([true_map[c] for c in candidate_ids], dtype=np.float64)
        # Missing scores default to 0.0 (model must score every candidate; this
        # is a safety net, not an expected path).
        pred = np.array([score_map.get(c, 0.0) for c in candidate_ids], dtype=np.float64)
        if len(candidate_ids) == 0:
            continue
        spearmans.append(spearman(pred, true))
        pa = pairwise_accuracy(pred, true)
        if pa is not None:
            pair_accs.append(pa)
        chosen = _selected_index(pred)
        best_true = float(true.max())
        selected_utils.append(float(true[chosen]))
        regrets.append(best_true - float(true[chosen]))

    return {
        "spearman": float(np.mean(spearmans)) if spearmans else 0.0,
        "pair_acc": float(np.mean(pair_accs)) if pair_accs else 0.0,
        "top1_regret": float(np.mean(regrets)) if regrets else 0.0,
        "mean_true_utility": float(np.mean(selected_utils)) if selected_utils else 0.0,
    }


def average_directions(per_direction: List[Dict[str, float]]) -> Dict[str, float]:
    """Average a list of per-direction metric dicts."""
    keys = per_direction[0].keys()
    return {k: float(np.mean([d[k] for d in per_direction])) for k in keys}


def add_lift(
    model_metrics: Dict[str, float],
    reference_mtu: Dict[str, float],
) -> Dict[str, float]:
    """Attach lift_random/lift_reward/lift_prob given reference mean-true-utility."""
    mtu = model_metrics["mean_true_utility"]
    out = dict(model_metrics)
    out["lift_random"] = mtu - reference_mtu["random"]
    out["lift_reward"] = mtu - reference_mtu["reward"]
    out["lift_prob"] = mtu - reference_mtu["prob"]
    return out
