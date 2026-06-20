"""TAP v1 dataset loading, joining, chain splits, and train-only standardization.

Reads the four schema-valid Parquet files (states/trajectories/candidates/
history) produced by :mod:`tap.synth` (synthetic) or a real collection run, and
exposes everything the model/baseline/eval waves need:

* :func:`load_parquets`     -> :class:`TapData` (raw frames + a candidate frame
  left-joined to its state's ``*_before`` and numeric state columns per the
  utility contract).
* :func:`chain_splits`      -> the two leave-one-chain-out directions
  (train chain 0 / test chain 1, then the swap). Every candidate of a state
  stays together automatically because ``chain_id`` is a state-level attribute.
  With >=128 labels, the last two states (by step) of each training chain are
  held out for early stopping.
* :class:`Standardizer`     -> fit mean/std on TRAIN ONLY, transform anything.

The canonical feature-column groups live here too so :mod:`tap.featurize`,
:mod:`tap.baselines`, and :mod:`tap.train` all agree on which columns are
features, which are probability/gradient/history blocks (for ablations), and
which are LEAK columns derived from the label.

Pure CPU: only numpy/pandas/pyarrow. No torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from tap.schema import FILE_NAMES, validate_parquet_dir

# --- The training label -----------------------------------------------------
LABEL_COL = "utility_points"

# Columns derived from the label (probe-after / gains / exact-match). These must
# NEVER be used as features — they leak the target.
LEAK_COLS = (
    "matched_probe_nll_after",
    "global_probe_nll_after",
    "generic_kl_after",
    "matched_gain",
    "global_gain",
    "incremental_generic_kl",
    "utility_points",
    "matched_exact_match_before",
    "matched_exact_match_after",
)

# --- Candidate numeric scalar features (safe, non-leak) ---------------------
CANDIDATE_NUMERIC_COLS = (
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
    "candidate_log_probability_change",
)

# Probability/familiarity columns (zeroed for the no-prob ablation).
PROB_COLS = (
    "candidate_mean_log_probability",
    "candidate_geometric_mean_probability",
    "candidate_arithmetic_mean_probability",
)

# Gradient-feature columns (zeroed, together with the gradient_sketch vector,
# for the no-grad ablation).
GRAD_SCALAR_COLS = (
    "gradient_norm",
    "estimated_update_norm",
    "max_gradient_similarity_to_history",
    "mean_gradient_similarity_to_history",
)

# Candidate-level "similarity to history" scalars (zeroed, together with the
# history block, for the no-history ablation).
HISTORY_SIM_COLS = (
    "max_semantic_similarity_to_history",
    "mean_semantic_similarity_to_history",
    "max_gradient_similarity_to_history",
    "mean_gradient_similarity_to_history",
)

# --- State numeric scalar features (joined onto each candidate) -------------
STATE_NUMERIC_COLS = (
    "step",
    "learning_rate",
    "grpo_beta",
    "clip_range",
    "lora_rank",
    "matched_probe_nll_before",
    "global_probe_nll_before",
    "generic_kl_before",
    "adam_first_moment_norm",
    "adam_second_moment_norm",
)

# --- History per-record numeric features (relative_age handled separately) --
HISTORY_NUMERIC_COLS = (
    "historical_reward_mean",
    "historical_advantage_mean",
    "historical_mean_log_probability",
    "historical_mean_entropy",
    "historical_update_norm",
    "historical_training_loss_change",
    "historical_candidate_log_probability_change",
)

HISTORY_WINDOW = 4  # latest four applied update batches (spec)
EARLY_STOP_MIN_LABELS = 128  # hold out last 2 states/chain only at >= this many


@dataclass
class TapData:
    """Loaded TAP frames plus a candidate frame joined to its state."""

    states: pd.DataFrame
    trajectories: pd.DataFrame
    candidates: pd.DataFrame
    history: pd.DataFrame
    joined: pd.DataFrame  # candidates LEFT JOIN state numeric/_before columns

    @property
    def n_labels(self) -> int:
        return len(self.candidates)

    @property
    def chains(self) -> List[str]:
        return sorted(self.states["chain_id"].unique().tolist())


def join_states(states: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Left-join state numeric/_before columns + policy_fingerprint onto candidates.

    Idempotent: if ``candidates`` already carries the joined columns it is
    returned unchanged (so ``score()`` works on either raw or joined frames).
    candidates already carry state_id/chain_id/step, so ``step`` is taken from
    the candidate side.
    """
    have = set(candidates.columns)
    needed = {"policy_fingerprint", *(c for c in STATE_NUMERIC_COLS if c != "step")}
    if needed.issubset(have):
        return candidates
    state_cols = ["state_id", "policy_fingerprint", *[c for c in STATE_NUMERIC_COLS if c != "step"]]
    state_view = states[state_cols].copy()
    joined = candidates.merge(state_view, on="state_id", how="left", validate="many_to_one")
    return joined.reset_index(drop=True)


def load_parquets(parquet_dir: str | Path, validate: bool = True) -> TapData:
    """Load the four Parquet files and build the joined candidate frame."""
    directory = Path(parquet_dir)
    if validate:
        validate_parquet_dir(directory)
    frames = {
        name.replace(".parquet", ""): pd.read_parquet(directory / name)
        for name in FILE_NAMES
    }
    states = frames["states"]
    candidates = frames["candidates"]
    trajectories = frames["trajectories"]
    history = frames["history"]

    joined = join_states(states, candidates)

    return TapData(
        states=states,
        trajectories=trajectories,
        candidates=candidates,
        history=history,
        joined=joined,
    )


@dataclass
class ChainSplit:
    """One leave-one-chain-out direction."""

    direction: int
    train_chain: str
    test_chain: str
    train: pd.DataFrame  # all train-chain candidates (joined rows)
    test: pd.DataFrame  # all test-chain candidates (joined rows)
    fit: pd.DataFrame  # train minus early-stop holdout (== train when not used)
    val: pd.DataFrame  # early-stop holdout (empty unless >=128 labels)

    @property
    def train_state_ids(self) -> set:
        return set(self.train["state_id"].unique())

    @property
    def test_state_ids(self) -> set:
        return set(self.test["state_id"].unique())


def _last_two_state_ids(frame: pd.DataFrame) -> set:
    """The last two states (by step, then state_id) of a single-chain frame."""
    order = (
        frame[["state_id", "step"]]
        .drop_duplicates()
        .sort_values(["step", "state_id"])
    )
    return set(order["state_id"].tolist()[-2:])


def chain_splits(data: TapData) -> List[ChainSplit]:
    """Return both leave-one-chain-out directions (never a random split).

    Direction 0: train chain[0], test chain[1].
    Direction 1: train chain[1], test chain[0].

    With >=128 labels, the last two states (by step) of the training chain are
    moved into ``val`` for early stopping; ``fit`` is the remainder.
    """
    joined = data.joined
    chains = data.chains
    if len(chains) != 2:
        raise ValueError(f"chain_splits expects exactly 2 chains, got {chains}")
    use_early_stop = data.n_labels >= EARLY_STOP_MIN_LABELS

    splits: List[ChainSplit] = []
    for direction, (train_chain, test_chain) in enumerate(
        [(chains[0], chains[1]), (chains[1], chains[0])]
    ):
        train = joined[joined["chain_id"] == train_chain].reset_index(drop=True)
        test = joined[joined["chain_id"] == test_chain].reset_index(drop=True)
        if use_early_stop:
            val_ids = _last_two_state_ids(train)
            val = train[train["state_id"].isin(val_ids)].reset_index(drop=True)
            fit = train[~train["state_id"].isin(val_ids)].reset_index(drop=True)
        else:
            val = train.iloc[0:0].copy()
            fit = train.copy()
        splits.append(
            ChainSplit(
                direction=direction,
                train_chain=str(train_chain),
                test_chain=str(test_chain),
                train=train,
                test=test,
                fit=fit,
                val=val,
            )
        )
    return splits


class Standardizer:
    """Column-wise standardizer fit on TRAIN ONLY.

    ``fit`` records per-column mean/std; ``transform`` applies ``(x-mean)/std``.
    Zero-variance columns (e.g. constant grpo_beta/lora_rank) use std=1 so they
    map to 0 instead of NaN.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < self.eps] = 1.0
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer.transform called before fit")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
