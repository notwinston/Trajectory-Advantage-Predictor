"""Feature assembly + ablation masks for TAP v1.

Two consumers:

* sklearn baselines (ridge / gbt / no-history MLP / numeric-only / candidate-only)
  use :func:`build_flat`, a single flat numeric matrix per candidate built from
  named blocks, with per-baseline *views* that include/exclude blocks.
* :class:`tap.model.SmallTAP` uses :func:`build_tap_blocks`, the structured
  per-block tensors (candidate embedding, gradient sketch, candidate numerics,
  state numerics, policy fingerprint, and a padded ``[N, 8, rec_dim]`` history
  tensor with a mask and relative-step indices).

Ablations (no-prob / no-grad / no-history) zero the corresponding columns/blocks
so a single model definition covers TAP and its three ablations.

Raw features only — standardization is fit on TRAIN ONLY downstream
(:class:`tap.dataset.Standardizer`). Pure CPU: numpy/pandas. No torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tap.dataset import (
    CANDIDATE_NUMERIC_COLS,
    GRAD_SCALAR_COLS,
    HISTORY_NUMERIC_COLS,
    HISTORY_SIM_COLS,
    HISTORY_WINDOW,
    LABEL_COL,
    PROB_COLS,
    STATE_NUMERIC_COLS,
)
from tap.schema import (
    CANDIDATE_EMBEDDING_DIM,
    GRADIENT_SKETCH_DIM,
    POLICY_FINGERPRINT_DIM,
)

# History per-record feature dim for the TAP history tensor:
#   historical_candidate_embedding (256) + historical_gradient_sketch (64)
#   + HISTORY_NUMERIC_COLS (7). relative_age is carried separately as an index.
HISTORY_RECORD_DIM = CANDIDATE_EMBEDDING_DIM + GRADIENT_SKETCH_DIM + len(HISTORY_NUMERIC_COLS)

# Flat-matrix block dims (history_agg = n_history + mean_relative_age + mean of
# the 7 history numerics).
HISTORY_AGG_DIM = 2 + len(HISTORY_NUMERIC_COLS)

FLAT_BLOCK_DIMS: Dict[str, int] = {
    "cand_numeric": len(CANDIDATE_NUMERIC_COLS),
    "state_numeric": len(STATE_NUMERIC_COLS),
    "cand_emb": CANDIDATE_EMBEDDING_DIM,
    "grad_sketch": GRADIENT_SKETCH_DIM,
    "fingerprint": POLICY_FINGERPRINT_DIM,
    "history_agg": HISTORY_AGG_DIM,
}
ALL_FLAT_BLOCKS = tuple(FLAT_BLOCK_DIMS.keys())

# Per-baseline flat views: which blocks to include + which columns to zero.
VIEWS: Dict[str, Dict] = {
    "full": dict(blocks=ALL_FLAT_BLOCKS, zero_cols=()),
    # no-history MLP: drop the history aggregate block and zero the candidate's
    # similarity-to-history scalars.
    "no_history": dict(
        blocks=tuple(b for b in ALL_FLAT_BLOCKS if b != "history_agg"),
        zero_cols=HISTORY_SIM_COLS,
    ),
    # numeric-only: scalar numerics, no learned embeddings/sketch/fingerprint.
    "numeric_only": dict(blocks=("cand_numeric", "state_numeric"), zero_cols=()),
    # candidate-only: only candidate-side blocks (no state, no fingerprint, no history).
    "candidate_only": dict(blocks=("cand_numeric", "cand_emb", "grad_sketch"), zero_cols=()),
}


def _vec_matrix(frame: pd.DataFrame, col: str, dim: int) -> np.ndarray:
    """Stack a list<float32> column into an ``[N, dim]`` float64 matrix."""
    out = np.zeros((len(frame), dim), dtype=np.float64)
    for i, v in enumerate(frame[col].to_numpy()):
        arr = np.asarray(v, dtype=np.float64)
        out[i, : arr.shape[0]] = arr
    return out


def _scalar_matrix(frame: pd.DataFrame, cols) -> np.ndarray:
    return frame[list(cols)].to_numpy(dtype=np.float64)


def _history_lookup(history_df: pd.DataFrame) -> Dict[str, List[dict]]:
    """state_id -> history records ordered by history_position (most-recent first)."""
    lookup: Dict[str, List[dict]] = {}
    if len(history_df) == 0:
        return lookup
    ordered = history_df.sort_values(["state_id", "history_position"])
    for state_id, group in ordered.groupby("state_id", sort=False):
        lookup[state_id] = group.to_dict("records")
    return lookup


# --------------------------------------------------------------------------- #
# Flat features (sklearn baselines)
# --------------------------------------------------------------------------- #
def _history_agg_matrix(frame: pd.DataFrame, history_df: pd.DataFrame) -> np.ndarray:
    lookup = _history_lookup(history_df)
    out = np.zeros((len(frame), HISTORY_AGG_DIM), dtype=np.float64)
    for i, state_id in enumerate(frame["state_id"].to_numpy()):
        recs = lookup.get(state_id, [])
        n = len(recs)
        out[i, 0] = float(n)
        if n == 0:
            continue
        out[i, 1] = float(np.mean([r["relative_age"] for r in recs]))
        for j, col in enumerate(HISTORY_NUMERIC_COLS):
            out[i, 2 + j] = float(np.mean([r[col] for r in recs]))
    return out


def _block_matrix(frame: pd.DataFrame, history_df: pd.DataFrame, block: str) -> Tuple[np.ndarray, List[str]]:
    if block == "cand_numeric":
        return _scalar_matrix(frame, CANDIDATE_NUMERIC_COLS), list(CANDIDATE_NUMERIC_COLS)
    if block == "state_numeric":
        return _scalar_matrix(frame, STATE_NUMERIC_COLS), list(STATE_NUMERIC_COLS)
    if block == "cand_emb":
        m = _vec_matrix(frame, "candidate_embedding", CANDIDATE_EMBEDDING_DIM)
        return m, [f"cand_emb_{i}" for i in range(CANDIDATE_EMBEDDING_DIM)]
    if block == "grad_sketch":
        m = _vec_matrix(frame, "gradient_sketch", GRADIENT_SKETCH_DIM)
        return m, [f"grad_sketch_{i}" for i in range(GRADIENT_SKETCH_DIM)]
    if block == "fingerprint":
        m = _vec_matrix(frame, "policy_fingerprint", POLICY_FINGERPRINT_DIM)
        return m, [f"fingerprint_{i}" for i in range(POLICY_FINGERPRINT_DIM)]
    if block == "history_agg":
        m = _history_agg_matrix(frame, history_df)
        names = ["history_count", "history_mean_relative_age"] + [
            f"history_mean_{c}" for c in HISTORY_NUMERIC_COLS
        ]
        return m, names
    raise ValueError(f"unknown flat block: {block!r}")


def build_flat(
    frame: pd.DataFrame, history_df: pd.DataFrame, view: str = "full"
) -> Tuple[np.ndarray, List[str]]:
    """Build a flat ``[N, D]`` feature matrix for a baseline ``view``."""
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; known: {sorted(VIEWS)}")
    cfg = VIEWS[view]
    mats: List[np.ndarray] = []
    names: List[str] = []
    for block in cfg["blocks"]:
        m, n = _block_matrix(frame, history_df, block)
        mats.append(m)
        names.extend(n)
    X = np.concatenate(mats, axis=1) if mats else np.zeros((len(frame), 0))
    # zero out requested columns (ablation-style) by name.
    zero_cols = set(cfg.get("zero_cols", ()))
    if zero_cols:
        for idx, name in enumerate(names):
            if name in zero_cols:
                X[:, idx] = 0.0
    return X, names


def labels(frame: pd.DataFrame) -> np.ndarray:
    return frame[LABEL_COL].to_numpy(dtype=np.float64)


# --------------------------------------------------------------------------- #
# Structured blocks (SmallTAP)
# --------------------------------------------------------------------------- #
@dataclass
class TapBlocks:
    cand_emb: np.ndarray  # [N, 256]
    grad_sketch: np.ndarray  # [N, 64]
    cand_numeric: np.ndarray  # [N, 16]
    state_numeric: np.ndarray  # [N, 10]
    fingerprint: np.ndarray  # [N, POLICY_FINGERPRINT_DIM]
    history: np.ndarray  # [N, HISTORY_WINDOW, HISTORY_RECORD_DIM]
    history_mask: np.ndarray  # [N, HISTORY_WINDOW] bool (True = real record)
    history_rel_age: np.ndarray  # [N, HISTORY_WINDOW] int (0 padding, else relative age)
    state_ids: np.ndarray  # [N] str
    candidate_ids: np.ndarray  # [N] str
    y: np.ndarray  # [N] utility_points

    def __len__(self) -> int:
        return self.cand_emb.shape[0]


# Column indices used by ablation masks.
_PROB_IDX = [CANDIDATE_NUMERIC_COLS.index(c) for c in PROB_COLS]
_GRAD_SCALAR_IDX = [CANDIDATE_NUMERIC_COLS.index(c) for c in GRAD_SCALAR_COLS]
_HISTORY_SIM_IDX = [CANDIDATE_NUMERIC_COLS.index(c) for c in HISTORY_SIM_COLS]


def build_tap_blocks(
    frame: pd.DataFrame, history_df: pd.DataFrame, ablation: str | None = None
) -> TapBlocks:
    """Build SmallTAP input blocks; ``ablation`` in {None, no_prob, no_grad, no_history}."""
    n = len(frame)
    cand_emb = _vec_matrix(frame, "candidate_embedding", CANDIDATE_EMBEDDING_DIM)
    grad_sketch = _vec_matrix(frame, "gradient_sketch", GRADIENT_SKETCH_DIM)
    cand_numeric = _scalar_matrix(frame, CANDIDATE_NUMERIC_COLS)
    state_numeric = _scalar_matrix(frame, STATE_NUMERIC_COLS)
    fingerprint = _vec_matrix(frame, "policy_fingerprint", POLICY_FINGERPRINT_DIM)

    history = np.zeros((n, HISTORY_WINDOW, HISTORY_RECORD_DIM), dtype=np.float64)
    history_mask = np.zeros((n, HISTORY_WINDOW), dtype=bool)
    history_rel_age = np.zeros((n, HISTORY_WINDOW), dtype=np.int64)
    lookup = _history_lookup(history_df)
    emb_dim = CANDIDATE_EMBEDDING_DIM
    grad_dim = GRADIENT_SKETCH_DIM
    for i, state_id in enumerate(frame["state_id"].to_numpy()):
        recs = lookup.get(state_id, [])[:HISTORY_WINDOW]
        for pos, rec in enumerate(recs):
            emb = np.asarray(rec["historical_candidate_embedding"], dtype=np.float64)
            grad = np.asarray(rec["historical_gradient_sketch"], dtype=np.float64)
            nums = np.asarray([rec[c] for c in HISTORY_NUMERIC_COLS], dtype=np.float64)
            history[i, pos, :emb_dim] = emb
            history[i, pos, emb_dim : emb_dim + grad_dim] = grad
            history[i, pos, emb_dim + grad_dim :] = nums
            history_mask[i, pos] = True
            history_rel_age[i, pos] = int(rec["relative_age"])

    cand_numeric = cand_numeric.copy()
    if ablation == "no_prob":
        cand_numeric[:, _PROB_IDX] = 0.0
    elif ablation == "no_grad":
        cand_numeric[:, _GRAD_SCALAR_IDX] = 0.0
        grad_sketch = np.zeros_like(grad_sketch)
    elif ablation == "no_history":
        cand_numeric[:, _HISTORY_SIM_IDX] = 0.0
        history = np.zeros_like(history)
        history_mask = np.zeros_like(history_mask)
        history_rel_age = np.zeros_like(history_rel_age)
    elif ablation is not None:
        raise ValueError(f"unknown ablation {ablation!r}")

    return TapBlocks(
        cand_emb=cand_emb,
        grad_sketch=grad_sketch,
        cand_numeric=cand_numeric,
        state_numeric=state_numeric,
        fingerprint=fingerprint,
        history=history,
        history_mask=history_mask,
        history_rel_age=history_rel_age,
        state_ids=frame["state_id"].to_numpy(),
        candidate_ids=frame["candidate_id"].to_numpy(),
        y=labels(frame),
    )
