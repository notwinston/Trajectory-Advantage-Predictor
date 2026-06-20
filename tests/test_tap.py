"""Wave 1b tests — dataset splits, featurize masks, baselines, SmallTAP, eval.

Developed entirely against the synthetic Parquet from :mod:`tap.synth`. Runs on
CPU; the SmallTAP/torch tests fall back to the sklearn path when
``TAP_NO_TORCH=1`` is set (or torch is unavailable).
"""

import os
import tempfile
import unittest

import numpy as np

from tap import featurize as F
from tap.dataset import (
    CANDIDATE_NUMERIC_COLS,
    PROB_COLS,
    Standardizer,
    chain_splits,
    load_parquets,
)
from tap.synth import generate


def _synth_dir(stack, labels=72):
    tmp = stack.enter_context(tempfile.TemporaryDirectory())
    generate(tmp, labels=labels, chains=2, candidates_per_state=6, seed=1729)
    return tmp


class _SynthCase(unittest.TestCase):
    """Shared 72-label synthetic dataset (generated once per class)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        generate(cls._tmp.name, labels=72, chains=2, candidates_per_state=6, seed=1729)
        cls.data = load_parquets(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()


# --------------------------------------------------------------------------- #
# Phase 1 — dataset / chain splits / standardizer
# --------------------------------------------------------------------------- #
class ChainSplitTests(_SynthCase):
    def test_two_directions(self):
        splits = chain_splits(self.data)
        self.assertEqual(len(splits), 2)
        self.assertEqual(splits[0].train_chain, "0")
        self.assertEqual(splits[0].test_chain, "1")
        self.assertEqual(splits[1].train_chain, "1")
        self.assertEqual(splits[1].test_chain, "0")

    def test_train_test_chain_disjoint(self):
        for s in chain_splits(self.data):
            self.assertTrue(s.train_state_ids.isdisjoint(s.test_state_ids))
            self.assertEqual(set(s.train["chain_id"]), {s.train_chain})
            self.assertEqual(set(s.test["chain_id"]), {s.test_chain})

    def test_state_atomicity(self):
        # Every candidate of a state lands in exactly one side of the split.
        for s in chain_splits(self.data):
            train_states = set(s.train["state_id"])
            test_states = set(s.test["state_id"])
            self.assertTrue(train_states.isdisjoint(test_states))
            # all 6 candidates of each train state are present together.
            for state_id, grp in s.train.groupby("state_id"):
                self.assertEqual(len(grp), 6)

    def test_early_stop_only_at_128(self):
        # 72 labels -> no early-stop holdout.
        for s in chain_splits(self.data):
            self.assertEqual(len(s.val), 0)
            self.assertEqual(len(s.fit), len(s.train))

    def test_early_stop_holdout_at_128(self):
        with tempfile.TemporaryDirectory() as tmp:
            generate(tmp, labels=128, chains=2, candidates_per_state=8, seed=1729)
            data = load_parquets(tmp)
            for s in chain_splits(data):
                # last 2 states/chain (8 candidates each) held out for early stop.
                self.assertEqual(len(s.val), 16)
                self.assertEqual(len(s.fit), len(s.train) - 16)
                val_states = set(s.val["state_id"])
                fit_states = set(s.fit["state_id"])
                self.assertTrue(val_states.isdisjoint(fit_states))


class StandardizerTests(_SynthCase):
    def test_fit_on_train_only(self):
        splits = chain_splits(self.data)
        Xtr, _ = F.build_flat(splits[0].fit, self.data.history, "numeric_only")
        Xte, _ = F.build_flat(splits[0].test, self.data.history, "numeric_only")
        std = Standardizer().fit(Xtr)
        Ztr = std.transform(Xtr)
        # train columns standardize to ~0 mean / unit std (non-constant cols).
        self.assertTrue(np.allclose(Ztr.mean(axis=0), 0.0, atol=1e-6))
        Zte = std.transform(Xte)
        # test transform uses TRAIN stats -> test mean need not be 0.
        self.assertTrue(np.isfinite(Zte).all())

    def test_zero_variance_safe(self):
        X = np.ones((5, 3))
        Z = Standardizer().fit_transform(X)
        self.assertTrue(np.isfinite(Z).all())
        self.assertTrue(np.allclose(Z, 0.0))


# --------------------------------------------------------------------------- #
# Phase 2 — featurize blocks & ablation masks
# --------------------------------------------------------------------------- #
class FeaturizeTests(_SynthCase):
    def test_flat_view_shapes(self):
        n = self.data.n_labels
        expected = {
            "full": sum(F.FLAT_BLOCK_DIMS.values()),
            "numeric_only": F.FLAT_BLOCK_DIMS["cand_numeric"] + F.FLAT_BLOCK_DIMS["state_numeric"],
        }
        for view, dim in expected.items():
            X, names = F.build_flat(self.data.joined, self.data.history, view)
            self.assertEqual(X.shape, (n, dim))
            self.assertEqual(len(names), dim)
            self.assertTrue(np.isfinite(X).all())

    def test_no_history_view_drops_history_block(self):
        X, names = F.build_flat(self.data.joined, self.data.history, "no_history")
        self.assertFalse(any(name.startswith("history_") for name in names))
        # similarity-to-history scalars are zeroed.
        for col in ("max_semantic_similarity_to_history", "mean_gradient_similarity_to_history"):
            self.assertTrue((X[:, names.index(col)] == 0).all())

    def test_tap_block_shapes(self):
        b = F.build_tap_blocks(self.data.joined, self.data.history)
        n = self.data.n_labels
        self.assertEqual(b.cand_emb.shape, (n, 256))
        self.assertEqual(b.grad_sketch.shape, (n, 64))
        self.assertEqual(b.cand_numeric.shape, (n, len(CANDIDATE_NUMERIC_COLS)))
        self.assertEqual(b.state_numeric.shape, (n, 10))
        self.assertEqual(b.fingerprint.shape, (n, 16))
        self.assertEqual(b.history.shape, (n, 4, F.HISTORY_RECORD_DIM))
        self.assertEqual(b.history_mask.shape, (n, 4))
        self.assertTrue(b.history_mask.any())  # some states have history

    def test_ablation_masks(self):
        prob_idx = [CANDIDATE_NUMERIC_COLS.index(c) for c in PROB_COLS]
        b_prob = F.build_tap_blocks(self.data.joined, self.data.history, ablation="no_prob")
        self.assertTrue((b_prob.cand_numeric[:, prob_idx] == 0).all())
        b_grad = F.build_tap_blocks(self.data.joined, self.data.history, ablation="no_grad")
        self.assertTrue((b_grad.grad_sketch == 0).all())
        b_hist = F.build_tap_blocks(self.data.joined, self.data.history, ablation="no_history")
        self.assertFalse(b_hist.history_mask.any())
        self.assertTrue((b_hist.history == 0).all())


if __name__ == "__main__":
    unittest.main()
