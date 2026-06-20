"""Small TAP model construction and lightweight baseline training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from tap_loop.metrics import BASELINES, baseline_score, evaluate_ranker


NUMERIC_FEATURES = (
    "step",
    "learning_rate",
    "grpo_beta",
    "adam_first_moment_norm",
    "adam_second_moment_norm",
    "candidate_reward_mean",
    "candidate_reward_std",
    "candidate_advantage_mean",
    "candidate_advantage_std",
    "candidate_mean_log_probability",
    "candidate_geometric_mean_probability",
    "candidate_arithmetic_mean_probability",
    "candidate_mean_entropy",
    "candidate_mean_sequence_length",
    "gradient_norm",
    "estimated_update_norm",
    "max_semantic_similarity_to_history",
    "mean_semantic_similarity_to_history",
    "max_gradient_similarity_to_history",
    "mean_gradient_similarity_to_history",
)


@dataclass(frozen=True)
class RidgeModel:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    weights: np.ndarray

    def predict(self, rows: Sequence[dict]) -> np.ndarray:
        x = design_matrix(rows, mean=self.feature_mean, std=self.feature_std)
        return x @ self.weights + self.target_mean


def design_matrix(
    rows: Sequence[dict],
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    matrix = np.asarray(
        [[float(row.get(feature, 0.0) or 0.0) for feature in NUMERIC_FEATURES] for row in rows],
        dtype=np.float64,
    )
    if mean is None:
        mean = matrix.mean(axis=0) if len(matrix) else np.zeros(len(NUMERIC_FEATURES))
    if std is None:
        std = matrix.std(axis=0) if len(matrix) else np.ones(len(NUMERIC_FEATURES))
    std = np.where(std == 0.0, 1.0, std)
    normalized = (matrix - mean) / std
    return np.concatenate([normalized, np.ones((normalized.shape[0], 1))], axis=1)


def train_ridge(rows: Sequence[dict], *, alpha: float = 1.0) -> RidgeModel:
    if not rows:
        raise ValueError("cannot train ridge on empty rows")
    raw = np.asarray(
        [[float(row.get(feature, 0.0) or 0.0) for feature in NUMERIC_FEATURES] for row in rows],
        dtype=np.float64,
    )
    mean = raw.mean(axis=0)
    std = np.where(raw.std(axis=0) == 0.0, 1.0, raw.std(axis=0))
    x = design_matrix(rows, mean=mean, std=std)
    y = np.asarray([float(row["utility_points"]) for row in rows], dtype=np.float64)
    target_mean = float(y.mean())
    centered = y - target_mean
    regularizer = alpha * np.eye(x.shape[1])
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(x.T @ x + regularizer, x.T @ centered)
    return RidgeModel(feature_mean=mean, feature_std=std, target_mean=target_mean, weights=weights)


def pairwise_ranking_loss(predicted: Sequence[float], target: Sequence[float], *, margin: float = 0.0) -> float:
    pred = np.asarray(predicted, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError("predicted and target must have the same shape")
    losses: list[float] = []
    for i in range(len(true)):
        for j in range(i + 1, len(true)):
            sign = np.sign(true[i] - true[j])
            if sign == 0:
                continue
            losses.append(float(np.logaddexp(0.0, -(pred[i] - pred[j]) * sign + margin)))
    return float(np.mean(losses)) if losses else 0.0


def chain_split_rows(rows: Iterable[dict], train_chain: int) -> tuple[list[dict], list[dict]]:
    train: list[dict] = []
    test: list[dict] = []
    for row in rows:
        if int(row["chain_id"]) == train_chain:
            train.append(row)
        else:
            test.append(row)
    if not train or not test:
        raise ValueError("chain split requires at least one train and one test row")
    return train, test


def evaluate_baselines(rows: Sequence[dict]) -> dict[str, dict[str, float]]:
    results = {name: evaluate_ranker(rows, lambda row, baseline=name: baseline_score(row, baseline)) for name in BASELINES}
    chain_ids = sorted({int(row["chain_id"]) for row in rows})
    if len(chain_ids) < 2:
        return results
    ridge_reports: list[dict[str, float]] = []
    for train_chain in chain_ids:
        train, test = chain_split_rows(rows, train_chain=train_chain)
        model = train_ridge(train)
        predictions = model.predict(test)
        by_id = {id(row): float(predictions[index]) for index, row in enumerate(test)}
        ridge_reports.append(evaluate_ranker(test, lambda row: by_id[id(row)]))
    if ridge_reports:
        keys = ridge_reports[0].keys()
        results["ridge"] = {key: float(np.mean([report[key] for report in ridge_reports])) for key in keys}
    return results


def build_tap_model(*, numeric_dim: int = len(NUMERIC_FEATURES)):
    """Build the small TAP torch model. Heavy deps are imported lazily."""

    import torch
    from torch import nn

    class SmallTAPModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.candidate_proj = nn.Sequential(nn.Linear(256, 64), nn.GELU())
            self.gradient_proj = nn.Sequential(nn.Linear(64, 32), nn.GELU())
            self.numeric_proj = nn.Sequential(nn.Linear(numeric_dim, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
            self.state_proj = nn.Sequential(nn.Linear(16, 32), nn.GELU())
            self.history_proj = nn.Sequential(nn.Linear(256 + 64 + 8, 64), nn.GELU())
            self.history_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(64 + 32 + 64 + 32 + 64, 128),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(128, 1),
            )

        def forward(self, numeric, state_fingerprint, candidate_embedding, gradient_sketch, history):
            candidate = self.candidate_proj(candidate_embedding)
            gradient = self.gradient_proj(gradient_sketch)
            numeric_hidden = self.numeric_proj(numeric)
            state = self.state_proj(state_fingerprint)
            history_hidden = self.history_proj(history)
            query = candidate.unsqueeze(1)
            attended, _ = self.history_attn(query, history_hidden, history_hidden)
            joined = torch.cat([candidate, gradient, numeric_hidden, state, attended.squeeze(1)], dim=-1)
            return self.head(joined).squeeze(-1)

    return SmallTAPModel()
