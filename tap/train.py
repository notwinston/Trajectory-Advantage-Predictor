"""Training + scoring wrapper for SmallTAP (and the TAP_NO_TORCH fallback).

``TapModel`` exposes the same ``fit(states, candidates, history) -> self`` /
``score(...) -> {state_id: {candidate_id: float}}`` contract as the baselines so
:mod:`tap.eval` / :mod:`tap.run_all` treat every model uniformly.

Loss (spec "TRAINING LOSS"): Huber on standardized ``utility_points`` + 0.5 *
within-state pairwise ranking loss + weight decay. Full-batch on the tiny (~36
train) dataset; epochs are capped and weight decay is strong (anti-overfit per
the spec's stuck-state guidance — TAP losing to a baseline is acceptable and
reported).

If ``TAP_NO_TORCH=1`` (or torch is unavailable), :func:`make_tap_model` returns a
scikit-learn ``MLPRegressor`` wrapper labelled "TAP v1 (simpler model)", which
the spec explicitly permits as TAP v1.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tap import featurize as F
from tap.baselines import ScoreDict, ScoringModel, _group_scores
from tap.dataset import (
    GRAD_SCALAR_COLS,
    HISTORY_SIM_COLS,
    PROB_COLS,
    Standardizer,
    join_states,
)
from tap.model import SmallTAP, torch_available

TORCH_LABEL = "TAP v1 (SmallTAP attention)"
FALLBACK_LABEL = "TAP v1 (simpler model)"


# --------------------------------------------------------------------------- #
# torch path
# --------------------------------------------------------------------------- #
class TapModel(ScoringModel):
    """SmallTAP trained with Huber + 0.5 pairwise ranking + weight decay."""

    trainable = True
    simpler = False
    label = TORCH_LABEL

    def __init__(
        self,
        name: str = "tap",
        ablation: Optional[str] = None,
        epochs: int = 300,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        pairwise_weight: float = 0.5,
        seed: int = 0,
    ):
        self.name = name
        self.ablation = ablation
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.pairwise_weight = pairwise_weight
        self.seed = seed
        self._model = None
        self._cand_scaler: Standardizer | None = None
        self._state_scaler: Standardizer | None = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self.history_ = {"initial_loss": None, "final_loss": None}

    # -- block tensors ----------------------------------------------------- #
    def _make_batch(self, blocks, torch):
        cand_numeric = self._cand_scaler.transform(blocks.cand_numeric)
        state_numeric = self._state_scaler.transform(blocks.state_numeric)

        def t(x, dtype=torch.float32):
            return torch.tensor(np.asarray(x), dtype=dtype)

        return {
            "cand_emb": t(blocks.cand_emb),
            "grad_sketch": t(blocks.grad_sketch),
            "cand_numeric": t(cand_numeric),
            "state_numeric": t(state_numeric),
            "fingerprint": t(blocks.fingerprint),
            "history": t(blocks.history),
            "history_mask": t(blocks.history_mask, torch.bool),
            "history_rel_age": t(blocks.history_rel_age, torch.long),
        }

    @staticmethod
    def _state_pairs(state_ids: np.ndarray) -> List[np.ndarray]:
        groups: Dict[str, List[int]] = {}
        for i, s in enumerate(state_ids):
            groups.setdefault(str(s), []).append(i)
        return [np.array(v, dtype=np.int64) for v in groups.values() if len(v) > 1]

    def fit(self, states_df, candidates_df, history_df) -> "TapModel":
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        joined = join_states(states_df, candidates_df)
        blocks = F.build_tap_blocks(joined, history_df, ablation=self.ablation)
        self._cand_scaler = Standardizer().fit(blocks.cand_numeric)
        self._state_scaler = Standardizer().fit(blocks.state_numeric)
        y = blocks.y.astype(np.float64)
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) or 1.0
        y_std = torch.tensor((y - self._y_mean) / self._y_std, dtype=torch.float32)

        self._model = SmallTAP()
        batch = self._make_batch(blocks, torch)
        groups = self._state_pairs(blocks.state_ids)

        opt = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        huber = torch.nn.SmoothL1Loss()
        logsig = torch.nn.LogSigmoid()

        self._model.train()
        for epoch in range(self.epochs):
            opt.zero_grad()
            pred = self._model(batch)
            loss = huber(pred, y_std)
            # within-state pairwise ranking loss.
            pair_terms = []
            for idx in groups:
                ps = pred[idx]
                ys = y_std[idx]
                pd_diff = ps.unsqueeze(1) - ps.unsqueeze(0)
                yd_diff = ys.unsqueeze(1) - ys.unsqueeze(0)
                mask = yd_diff > 0
                if mask.any():
                    pair_terms.append(-logsig(pd_diff[mask]))
            if pair_terms:
                pairwise = torch.cat(pair_terms).mean()
                loss = loss + self.pairwise_weight * pairwise
            loss.backward()
            opt.step()
            if epoch == 0:
                self.history_["initial_loss"] = float(loss.detach())
        self.history_["final_loss"] = float(loss.detach())
        return self

    def score(self, states_df, candidates_df, history_df) -> ScoreDict:
        import torch

        if self._model is None:
            raise RuntimeError(f"{self.name}: score before fit")
        joined = join_states(states_df, candidates_df)
        blocks = F.build_tap_blocks(joined, history_df, ablation=self.ablation)
        batch = self._make_batch(blocks, torch)
        self._model.eval()
        with torch.no_grad():
            pred = self._model(batch).cpu().numpy().astype(np.float64)
        pred = np.where(np.isfinite(pred), pred, 0.0)
        return _group_scores(joined, pred)


# --------------------------------------------------------------------------- #
# TAP_NO_TORCH fallback
# --------------------------------------------------------------------------- #
def _ablation_zero_cols(ablation: Optional[str]) -> Tuple[str, ...]:
    if ablation == "no_prob":
        return PROB_COLS
    if ablation == "no_grad":
        return GRAD_SCALAR_COLS  # grad_sketch_* zeroed separately below
    if ablation == "no_history":
        return HISTORY_SIM_COLS
    return ()


class TapFallback(ScoringModel):
    """sklearn MLPRegressor TAP — labelled 'TAP v1 (simpler model)'."""

    trainable = True
    simpler = True
    label = FALLBACK_LABEL

    def __init__(self, name: str = "tap", ablation: Optional[str] = None, seed: int = 0):
        self.name = name
        self.ablation = ablation
        self.seed = seed
        self._scaler: Standardizer | None = None
        self._est = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self.history_ = {"initial_loss": None, "final_loss": None}

    def _features(self, joined, history_df) -> np.ndarray:
        # no-history ablation drops the history block entirely.
        view = "no_history" if self.ablation == "no_history" else "full"
        X, names = F.build_flat(joined, history_df, view)
        zero = set(_ablation_zero_cols(self.ablation))
        for idx, name in enumerate(names):
            if name in zero or (self.ablation == "no_grad" and name.startswith("grad_sketch_")):
                X[:, idx] = 0.0
        return X

    def fit(self, states_df, candidates_df, history_df) -> "TapFallback":
        from sklearn.neural_network import MLPRegressor

        joined = join_states(states_df, candidates_df)
        X = self._features(joined, history_df)
        y = F.labels(joined)
        self._scaler = Standardizer().fit(X)
        Z = self._scaler.transform(X)
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) or 1.0
        y_std = (y - self._y_mean) / self._y_std
        self._est = MLPRegressor(
            hidden_layer_sizes=(128, 64), alpha=1e-2, max_iter=3000, random_state=self.seed
        )
        self._est.fit(Z, y_std)
        self.history_["final_loss"] = float(getattr(self._est, "loss_", float("nan")))
        return self

    def score(self, states_df, candidates_df, history_df) -> ScoreDict:
        if self._est is None or self._scaler is None:
            raise RuntimeError(f"{self.name}: score before fit")
        joined = join_states(states_df, candidates_df)
        Z = self._scaler.transform(self._features(joined, history_df))
        pred = np.asarray(self._est.predict(Z), dtype=np.float64)
        pred = np.where(np.isfinite(pred), pred, 0.0)
        return _group_scores(joined, pred)


def make_tap_model(name: str = "tap", ablation: Optional[str] = None, **kw):
    """Return the torch TapModel, or the sklearn fallback under TAP_NO_TORCH."""
    if torch_available():
        return TapModel(name=name, ablation=ablation, **kw)
    return TapFallback(name=name, ablation=ablation, seed=kw.get("seed", 0))


# TAP + the three ablations, keyed by their canonical result names.
def make_tap_models(seed: int = 0, **kw) -> Dict[str, ScoringModel]:
    return {
        "tap": make_tap_model("tap", None, seed=seed, **kw),
        "tap-no-prob": make_tap_model("tap-no-prob", "no_prob", seed=seed, **kw),
        "tap-no-grad": make_tap_model("tap-no-grad", "no_grad", seed=seed, **kw),
        "tap-no-history": make_tap_model("tap-no-history", "no_history", seed=seed, **kw),
    }
