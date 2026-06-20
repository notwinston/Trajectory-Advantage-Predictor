"""Ranking metrics and simple baseline scoring for TAP v1."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Callable, Iterable

import numpy as np


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        j = index
        while j + 1 < len(order) and values[order[j + 1]] == values[order[index]]:
            j += 1
        rank = (index + j) / 2.0
        for k in range(index, j + 1):
            ranks[order[k]] = rank
        index = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) != len(y):
        raise ValueError("spearman inputs must have same length")
    if len(x) < 2:
        return 0.0
    rx = np.asarray(_rank(x), dtype=np.float64)
    ry = np.asarray(_rank(y), dtype=np.float64)
    if float(rx.std()) == 0.0 or float(ry.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def pairwise_accuracy(predicted: list[float], actual: list[float]) -> float:
    if len(predicted) != len(actual):
        raise ValueError("pairwise_accuracy inputs must have same length")
    correct = 0
    total = 0
    for i in range(len(actual)):
        for j in range(i + 1, len(actual)):
            delta_true = actual[i] - actual[j]
            if delta_true == 0:
                continue
            delta_pred = predicted[i] - predicted[j]
            correct += int(delta_pred * delta_true > 0)
            total += 1
    return correct / total if total else 0.0


def top_one_regret(predicted: list[float], actual: list[float]) -> float:
    if not predicted:
        return 0.0
    chosen = max(range(len(predicted)), key=lambda index: predicted[index])
    best = max(actual)
    return float(best - actual[chosen])


def group_by_state(rows: Iterable[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    return dict(grouped)


def evaluate_ranker(rows: Iterable[dict], score_fn: Callable[[dict], float]) -> dict[str, float]:
    grouped = group_by_state(rows)
    spearman_values: list[float] = []
    pairwise_values: list[float] = []
    regrets: list[float] = []
    selected_utilities: list[float] = []
    for state_rows in grouped.values():
        predicted = [float(score_fn(row)) for row in state_rows]
        actual = [float(row["utility_points"]) for row in state_rows]
        spearman_values.append(spearman(predicted, actual))
        pairwise_values.append(pairwise_accuracy(predicted, actual))
        regrets.append(top_one_regret(predicted, actual))
        chosen = max(range(len(predicted)), key=lambda index: predicted[index])
        selected_utilities.append(actual[chosen])
    return {
        "states": float(len(grouped)),
        "within_state_spearman": float(np.mean(spearman_values)) if spearman_values else 0.0,
        "pairwise_accuracy": float(np.mean(pairwise_values)) if pairwise_values else 0.0,
        "top_one_regret": float(np.mean(regrets)) if regrets else 0.0,
        "selected_utility": float(np.mean(selected_utilities)) if selected_utilities else 0.0,
    }


def baseline_score(row: dict, name: str) -> float:
    if name == "random":
        return float(row.get("candidate_index", 0) * 0.0)
    if name == "reward_mean":
        return float(row.get("candidate_reward_mean", 0.0))
    if name == "advantage_mean":
        return float(row.get("candidate_advantage_mean", 0.0))
    if name == "geometric_probability":
        return float(row.get("candidate_geometric_mean_probability", 0.0))
    if name == "arithmetic_probability":
        return float(row.get("candidate_arithmetic_mean_probability", 0.0))
    if name == "reward_times_surprisal":
        return float(row.get("candidate_reward_mean", 0.0)) * -float(row.get("candidate_mean_log_probability", 0.0))
    if name == "semantic_novelty":
        return -float(row.get("max_semantic_similarity_to_history", 0.0))
    if name == "gradient_norm":
        return float(row.get("gradient_norm", 0.0))
    if name == "matched_probe_gradient_alignment":
        return float(row.get("matched_probe_gradient_alignment", 0.0))
    raise KeyError(f"unknown baseline: {name}")


BASELINES = (
    "random",
    "reward_mean",
    "advantage_mean",
    "geometric_probability",
    "arithmetic_probability",
    "reward_times_surprisal",
    "semantic_novelty",
    "gradient_norm",
    "matched_probe_gradient_alignment",
)


def lift_over(selected_utility: float, reference_utility: float) -> float:
    if math.isclose(reference_utility, 0.0):
        return selected_utility
    return selected_utility - reference_utility

