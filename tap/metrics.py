"""Metrics for TAP — regression *and* selection quality.

The TAP critique was that absolute regression error is not the goal: the use case
is **selecting which update to apply**, so the headline metrics are *within-anchor*
ranking and top-k selection lift, with bootstrap CIs (because the experiment is
label-poor and underpowered if reported naively).

Pure numpy; imports/tests on a laptop.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _rank(x: np.ndarray) -> np.ndarray:
    # average ranks would be ideal; argsort-argsort is fine for tie-light data.
    return np.argsort(np.argsort(x)).astype(float)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2:
        return 0.0
    return pearson(_rank(a), _rank(b))


def rmse(pred: Sequence[float], truth: Sequence[float]) -> float:
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    if len(truth) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def sign_accuracy(pred: Sequence[float], truth: Sequence[float], deadband: float = 0.0) -> float:
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    if len(truth) == 0:
        return float("nan")
    st = np.where(truth > deadband, 1, np.where(truth < -deadband, -1, 0))
    sp = np.where(pred > deadband, 1, np.where(pred < -deadband, -1, 0))
    return float(np.mean(st == sp))


# ---- selection / ranking metrics (the headline) -------------------------------


def _by_group(groups: Sequence) -> dict:
    idx: dict = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    return idx


def within_group_spearman(pred, truth, groups) -> float:
    """Mean Spearman(pred, truth) computed *within* each anchor/state group."""

    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    vals = []
    for members in _by_group(groups).values():
        if len(members) >= 2:
            vals.append(spearman(pred[members], truth[members]))
    return float(np.mean(vals)) if vals else 0.0


def pairwise_ranking_accuracy(pred, truth, groups) -> float:
    """Fraction of within-group candidate pairs whose order pred gets right."""

    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    correct = total = 0
    for members in _by_group(groups).values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                if truth[i] == truth[j]:
                    continue
                total += 1
                correct += int((pred[i] - pred[j]) * (truth[i] - truth[j]) > 0)
    return correct / total if total else float("nan")


def top1_regret(pred, truth, groups) -> float:
    """Mean (best true utility in group − true utility of pred's top pick)."""

    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    regrets = []
    for members in _by_group(groups).values():
        m = np.asarray(members)
        chosen = m[int(np.argmax(pred[m]))]
        regrets.append(float(np.max(truth[m]) - truth[chosen]))
    return float(np.mean(regrets)) if regrets else float("nan")


def topk_mean_true(pred, truth, groups, k_frac: float = 0.25) -> float:
    """Mean true utility of the top-``k_frac`` candidates (by pred) within groups."""

    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    picked = []
    for members in _by_group(groups).values():
        m = np.asarray(members)
        k = max(1, int(round(len(m) * k_frac)))
        order = m[np.argsort(pred[m])[::-1]]
        picked.extend(truth[order[:k]].tolist())
    return float(np.mean(picked)) if picked else float("nan")


def selection_lift(pred, truth, groups, k_frac: float = 0.25) -> dict:
    """Top-k mean true utility for the predictor vs random vs oracle, per group.

    ``lift_over_random`` = predictor's top-k mean − the group-mean (= expected
    random pick). ``frac_of_oracle`` normalizes by the achievable best.
    """

    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    model = topk_mean_true(pred, truth, groups, k_frac)
    rnd, oracle = [], []
    for members in _by_group(groups).values():
        m = np.asarray(members)
        k = max(1, int(round(len(m) * k_frac)))
        rnd.append(float(np.mean(truth[m])))
        oracle.append(float(np.mean(np.sort(truth[m])[::-1][:k])))
    rnd_m = float(np.mean(rnd)) if rnd else float("nan")
    oracle_m = float(np.mean(oracle)) if oracle else float("nan")
    denom = oracle_m - rnd_m
    return {
        "topk_model": model,
        "topk_random": rnd_m,
        "topk_oracle": oracle_m,
        "lift_over_random": model - rnd_m,
        "frac_of_oracle": (model - rnd_m) / denom if denom and np.isfinite(denom) and denom != 0 else float("nan"),
        "k_frac": k_frac,
    }


# ---- bootstrap CIs (the experiment is small; never report a point estimate) ----


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    values = np.asarray(values, float)
    if len(values) == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    boots = np.array([stat(values[rng.integers(0, len(values), len(values))]) for _ in range(n_boot)])
    return {
        "point": float(stat(values)),
        "lo": float(np.quantile(boots, alpha / 2)),
        "hi": float(np.quantile(boots, 1 - alpha / 2)),
    }


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float], **kw) -> dict:
    """CI on mean(a − b) — for "is A's selection better than B's", paired by group."""

    a, b = np.asarray(a, float), np.asarray(b, float)
    return bootstrap_ci(a - b, **kw)
