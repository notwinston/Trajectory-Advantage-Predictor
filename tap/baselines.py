"""Baseline models + heuristic selectors for TAP v1.

Every model exposes the uniform contract::

    model.fit(states_df, candidates_df, history_df) -> self
    model.score(states_df, candidates_df, history_df) -> {state_id: {candidate_id: float}}

where higher score == better candidate. Heuristic selectors ignore ``fit``
(no-op) and read a single candidate column; learned baselines standardize their
features fitting on the TRAIN candidates only, then a regressor predicts
``utility_points`` (used purely as a within-state ranking signal).

Learned baselines
    ridge, gbt (gradient-boosted trees), no_history_mlp, numeric_only, candidate_only
Heuristic selectors
    random (seeded), reward_mean, advantage_mean, geo_mean_prob, arith_mean_prob,
    reward_x_surprisal, semantic_novelty, gradient_norm, gradient_alignment

Pure CPU: numpy/pandas; scikit-learn is used when installed, otherwise a small
NumPy ridge fallback keeps local smoke tests runnable. No torch.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Dict

import numpy as np
import pandas as pd

from tap import featurize as F
from tap.dataset import Standardizer, join_states

ScoreDict = Dict[str, Dict[str, float]]


def _group_scores(candidates_df: pd.DataFrame, values: np.ndarray) -> ScoreDict:
    """Pack a per-row score array into {state_id: {candidate_id: float}}."""
    out: ScoreDict = {}
    state_ids = candidates_df["state_id"].to_numpy()
    candidate_ids = candidates_df["candidate_id"].to_numpy()
    for state_id, candidate_id, value in zip(state_ids, candidate_ids, values):
        out.setdefault(str(state_id), {})[str(candidate_id)] = float(value)
    return out


class ScoringModel:
    """Base class; heuristics use the default no-op fit."""

    name: str = "model"
    trainable: bool = False

    def fit(self, states_df, candidates_df, history_df) -> "ScoringModel":
        return self

    def score(self, states_df, candidates_df, history_df) -> ScoreDict:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Learned baselines
# --------------------------------------------------------------------------- #
class SklearnRegressor(ScoringModel):
    """A scikit-learn regressor on a flat featurize view, standardized on train."""

    trainable = True

    def __init__(self, name: str, view: str, estimator_factory: Callable):
        self.name = name
        self.view = view
        self._make = estimator_factory
        self._scaler: Standardizer | None = None
        self._est = None
        self._y_mean = 0.0
        self._y_std = 1.0

    def fit(self, states_df, candidates_df, history_df) -> "SklearnRegressor":
        joined = join_states(states_df, candidates_df)
        X, _ = F.build_flat(joined, history_df, self.view)
        y = F.labels(joined)
        self._scaler = Standardizer().fit(X)
        Z = self._scaler.transform(X)
        # standardize the target too (stabilizes ridge/MLP); ranking is scale-free.
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) or 1.0
        y_std = (y - self._y_mean) / self._y_std
        self._est = self._make()
        self._est.fit(Z, y_std)
        return self

    def score(self, states_df, candidates_df, history_df) -> ScoreDict:
        if self._est is None or self._scaler is None:
            raise RuntimeError(f"{self.name}: score before fit")
        joined = join_states(states_df, candidates_df)
        X, _ = F.build_flat(joined, history_df, self.view)
        Z = self._scaler.transform(X)
        pred = np.asarray(self._est.predict(Z), dtype=np.float64)
        pred = np.where(np.isfinite(pred), pred, 0.0)
        return _group_scores(joined, pred)


class _NumpyRidge:
    """Minimal ridge regressor fallback for environments without scikit-learn."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NumpyRidge":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        design = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        penalty = self.alpha * np.eye(design.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        lhs = design.T @ design + penalty
        rhs = design.T @ y
        try:
            self.coef_ = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.pinv(lhs) @ rhs
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("predict before fit")
        X = np.asarray(X, dtype=np.float64)
        design = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        return design @ self.coef_


def _ridge_factory():
    try:
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0)
    except ModuleNotFoundError:
        return _NumpyRidge(alpha=1.0)


def _gbt_factory():
    try:
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.9, random_state=0
        )
    except ModuleNotFoundError:
        return _NumpyRidge(alpha=0.5)


def _mlp_factory():
    try:
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-2,
            max_iter=2000,
            random_state=0,
        )
    except ModuleNotFoundError:
        return _NumpyRidge(alpha=0.25)


# --------------------------------------------------------------------------- #
# Heuristic selectors
# --------------------------------------------------------------------------- #
class HeuristicSelector(ScoringModel):
    """Stateless selector: score = ``column_fn(candidate_row)`` (higher better)."""

    def __init__(self, name: str, column_fn: Callable[[pd.DataFrame], np.ndarray]):
        self.name = name
        self._fn = column_fn

    def score(self, states_df, candidates_df, history_df) -> ScoreDict:
        joined = join_states(states_df, candidates_df)
        values = np.asarray(self._fn(joined), dtype=np.float64)
        values = np.where(np.isfinite(values), values, 0.0)
        return _group_scores(joined, values)


def _seeded_random(seed: int) -> Callable[[pd.DataFrame], np.ndarray]:
    def fn(df: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(df), dtype=np.float64)
        for i, cid in enumerate(df["candidate_id"].to_numpy()):
            h = hashlib.sha256(f"{seed}:{cid}".encode()).hexdigest()
            out[i] = int(h[:8], 16) / 0xFFFFFFFF
        return out

    return fn


def _col(name: str, sign: float = 1.0) -> Callable[[pd.DataFrame], np.ndarray]:
    return lambda df: sign * df[name].to_numpy(dtype=np.float64)


def _reward_x_surprisal(df: pd.DataFrame) -> np.ndarray:
    reward = df["candidate_reward_mean"].to_numpy(dtype=np.float64)
    surprisal = -df["candidate_mean_log_probability"].to_numpy(dtype=np.float64)
    return reward * surprisal


def _semantic_novelty(df: pd.DataFrame) -> np.ndarray:
    return 1.0 - df["max_semantic_similarity_to_history"].to_numpy(dtype=np.float64)


def make_baselines(seed: int = 0) -> Dict[str, ScoringModel]:
    """All learned + heuristic baselines, keyed by their canonical result name."""
    return {
        "ridge": SklearnRegressor("ridge", "full", _ridge_factory),
        "gbt": SklearnRegressor("gbt", "full", _gbt_factory),
        "no_history_mlp": SklearnRegressor("no_history_mlp", "no_history", _mlp_factory),
        "numeric_only": SklearnRegressor("numeric_only", "numeric_only", _ridge_factory),
        "candidate_only": SklearnRegressor("candidate_only", "candidate_only", _ridge_factory),
        "random": HeuristicSelector("random", _seeded_random(seed)),
        "reward_mean": HeuristicSelector("reward_mean", _col("candidate_reward_mean")),
        "advantage_mean": HeuristicSelector("advantage_mean", _col("candidate_advantage_mean")),
        "geo_mean_prob": HeuristicSelector(
            "geo_mean_prob", _col("candidate_geometric_mean_probability")
        ),
        "arith_mean_prob": HeuristicSelector(
            "arith_mean_prob", _col("candidate_arithmetic_mean_probability")
        ),
        "reward_x_surprisal": HeuristicSelector("reward_x_surprisal", _reward_x_surprisal),
        "semantic_novelty": HeuristicSelector("semantic_novelty", _semantic_novelty),
        "gradient_norm": HeuristicSelector("gradient_norm", _col("gradient_norm")),
        "gradient_alignment": HeuristicSelector(
            "gradient_alignment", _col("max_gradient_similarity_to_history")
        ),
    }


HEURISTIC_NAMES = (
    "random",
    "reward_mean",
    "advantage_mean",
    "geo_mean_prob",
    "arith_mean_prob",
    "reward_x_surprisal",
    "semantic_novelty",
    "gradient_norm",
    "gradient_alignment",
)
LEARNED_NAMES = ("ridge", "gbt", "no_history_mlp", "numeric_only", "candidate_only")
