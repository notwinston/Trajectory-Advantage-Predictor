"""Laptop unit tests for the TAP pure cores (no torch / no GPU)."""

from __future__ import annotations

import random
import unittest

from tap import metrics as M
from tap.cohorts import (
    Cohort, build_all_cohorts, duplication_cohorts, label_noise_cohorts, variance_decoupled_pair,
)
from tap.features import RolloutStats, summarize_rollouts, target_similarity
from tap.labels import LiftLabel, UtilityWeights, target_from_row
from tap import predictor as P
from tap import gate as G


def _rows(noise: float = 0.005, chains=3, anchors=3, cands=8, seeds=2, seed=0):
    rng = random.Random(seed)
    out = []
    for ch in range(chains):
        for a in range(anchors):
            fp = 1.5 - 0.1 * a
            for k in range(cands):
                fnd = rng.random(); red = rng.random(); ts = rng.random()
                nf = rng.choice([0.0, 0.0, 0.25, 1.0])
                true = 0.06 * fnd - 0.05 * red + 0.03 * ts - 0.07 * nf
                for s in range(seeds):
                    la = true + rng.gauss(0, noise)
                    out.append({
                        "chain_id": ch, "anchor_index": a, "candidate_id": f"{ch}-{a}-{k}", "seed": s,
                        "cohort": {"name": f"{ch}-{a}-{k}", "kind": "syn"},
                        "reward_summary": {
                            "reward_mean": 0.5, "reward_std": 0.5 * fnd, "pass_rate": 0.5,
                            "frac_nondegenerate": fnd, "frac_all_correct": 0.1, "frac_all_wrong": 0.1,
                            "redundancy_mean": red, "mean_logprob": -1 - red, "n_groups": 8,
                        },
                        "target_similarity": ts, "fingerprint_nll": fp, "fingerprint_entropy": 1.0,
                        "step_frac": a / max(anchors - 1, 1),
                        "lift_acc": la, "lift_nll": 4 * true, "kl_drift": 0.0, "utility": 100 * la,
                    })
    return out


class TestMetrics(unittest.TestCase):
    def test_spearman_monotone(self):
        self.assertAlmostEqual(M.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0, places=6)
        self.assertAlmostEqual(M.spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0, places=6)

    def test_within_group_and_regret(self):
        pred = [3, 2, 1, 3, 2, 1]
        truth = [3, 2, 1, 3, 2, 1]
        groups = ["a", "a", "a", "b", "b", "b"]
        self.assertAlmostEqual(M.within_group_spearman(pred, truth, groups), 1.0, places=6)
        self.assertEqual(M.top1_regret(pred, truth, groups), 0.0)
        self.assertAlmostEqual(M.pairwise_ranking_accuracy(pred, truth, groups), 1.0, places=6)

    def test_selection_lift_positive_when_pred_good(self):
        pred = [1, 2, 3, 1, 2, 3]
        truth = [1, 2, 3, 1, 2, 3]
        groups = ["a", "a", "a", "b", "b", "b"]
        sel = M.selection_lift(pred, truth, groups, k_frac=0.34)
        self.assertGreater(sel["lift_over_random"], 0.0)

    def test_bootstrap_ci_brackets_mean(self):
        ci = M.bootstrap_ci([1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(ci["point"], 1.0)
        self.assertLessEqual(ci["lo"], ci["point"])
        self.assertGreaterEqual(ci["hi"], ci["point"])


class TestLabels(unittest.TestCase):
    def test_lift_signs(self):
        lab = LiftLabel(acc_before=0.2, acc_after=0.3, nll_before=1.0, nll_after=0.8, kl_before=0.0, kl_after=0.1)
        self.assertAlmostEqual(lab.lift_acc, 0.1)
        self.assertAlmostEqual(lab.lift_nll, 0.2)
        self.assertAlmostEqual(lab.kl_drift, 0.1)

    def test_kl_drift_one_sided(self):
        lab = LiftLabel(0.2, 0.2, 1.0, 1.0, kl_before=0.5, kl_after=0.4)
        self.assertEqual(lab.kl_drift, 0.0)  # drift decreased -> no penalty

    def test_target_modes(self):
        row = {"lift_acc": 0.1, "lift_nll": 0.4, "kl_drift": 0.0}
        self.assertAlmostEqual(target_from_row(row, "acc"), 0.1)
        self.assertAlmostEqual(target_from_row(row, "nll"), 0.4)
        w = UtilityWeights(acc=1.0, nll=0.0, kl=0.05, scale=100.0)
        self.assertAlmostEqual(target_from_row(row, "utility", w), 10.0)


class TestFeatures(unittest.TestCase):
    def test_summarize_degenerate_vs_diverse(self):
        # group g1 all-correct (degenerate), g2 mixed (non-degenerate)
        rollouts = (
            [{"group_id": "g1", "reward": 1.0, "mean_logprob": -0.1, "completion_tokens": 10} for _ in range(4)]
            + [{"group_id": "g2", "reward": r, "mean_logprob": -1.0, "completion_tokens": 10} for r in (1.0, 0.0, 1.0, 0.0)]
        )
        s = summarize_rollouts(rollouts)
        self.assertEqual(s.n_groups, 2)
        self.assertAlmostEqual(s.frac_nondegenerate, 0.5)
        self.assertAlmostEqual(s.frac_all_correct, 0.5)
        self.assertIsNotNone(s.redundancy_mean)

    def test_target_similarity_self_is_high(self):
        from tap.features import _unigrams

        probe = _unigrams([[1, 2, 3, 4], [2, 3, 4, 5]])
        same = target_similarity([[1, 2, 3, 4], [2, 3, 4, 5]], probe)
        diff = target_similarity([[90, 91, 92]], probe)
        self.assertGreater(same, diff)


class TestCohorts(unittest.TestCase):
    def _pool(self, n=400):
        return [{"id": f"p{i}", "problem": "q", "answer": str(i), "level": f"Level {i % 5 + 1}",
                 "subject": ["Algebra", "Geometry"][i % 2]} for i in range(n)]

    def test_variance_decoupled_equal_mean_diff_var(self):
        rows = self._pool()
        pr = {f"p{i}": (0.5 if i % 3 == 0 else (0.05 if i % 3 == 1 else 0.95)) for i in range(len(rows))}
        hv, lv = variance_decoupled_pair(rows, pr, size=8)
        self.assertAlmostEqual(hv.meta["passrate_mean"], lv.meta["passrate_mean"], delta=0.2)
        self.assertGreater(hv.meta["within_group_var_mean"], lv.meta["within_group_var_mean"])

    def test_label_noise_counts(self):
        cs = label_noise_cohorts(self._pool(), size=8, fracs=(0.5,), max_per_frac=1)
        self.assertTrue(cs)
        self.assertEqual(len(cs[0].meta["noisy_ids"]), 4)

    def test_duplication_distinct(self):
        cs = duplication_cohorts(self._pool(), size=8, ks=(2,), max_per_k=1)
        self.assertEqual(len(set(cs[0].prompt_ids)), 2)
        self.assertEqual(len(cs[0].prompt_ids), 8)

    def test_build_all_runs(self):
        cs = build_all_cohorts(self._pool(), size=8)
        self.assertGreater(len(cs), 0)
        self.assertTrue(all(isinstance(c, Cohort) for c in cs))


class TestPredictor(unittest.TestCase):
    def test_build_xy_shapes(self):
        d = P.build_xy(_rows())
        self.assertEqual(d["X"].shape[0], len(d["y"]))
        self.assertIn("frac_nondegenerate", d["names"])
        self.assertIn("target_similarity", d["names"])

    def test_recovers_signal_and_beats_mean(self):
        rows = _rows(noise=0.003)
        rep = P.evaluate(rows, scheme="logo", backend="ridge", explain=False)
        # the learned model should rank within-anchor far better than predict-mean
        self.assertGreater(rep["within_anchor_spearman"], 0.3)
        self.assertGreater(rep["within_anchor_spearman"],
                           rep["baselines"]["predict_mean"]["within_anchor_spearman"])
        self.assertGreater(rep["selection"]["lift_over_random"], 0.0)

    def test_monotone_vector(self):
        names = ["frac_nondegenerate", "redundancy_mean", "pass_rate"]
        self.assertEqual(P.monotone_vector(names), [1, -1, 0])


class TestGate(unittest.TestCase):
    def test_low_noise_high_icc(self):
        rep = G.analyze(_rows(noise=0.001, seeds=3), label="acc")
        self.assertGreater(rep["icc"], 0.5)
        self.assertEqual(rep["verdict"], "strong_signal")

    def test_high_noise_low_icc(self):
        rep = G.analyze(_rows(noise=0.2, seeds=3), label="acc")
        self.assertLess(rep["icc"], 0.5)


class TestBatteryIO(unittest.TestCase):
    def test_completed_keys(self):
        import json as _json
        import os
        import tempfile

        from tap.battery import completed_keys  # imports w/o torch (lazy)

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for c, a, cand, s in [(0, 0, "x", 0), (0, 0, "x", 1), (1, 2, "y", 0)]:
                f.write(_json.dumps({"chain_id": c, "anchor_index": a, "candidate_id": cand, "seed": s}) + "\n")
            path = f.name
        try:
            self.assertEqual(completed_keys(path), {(0, 0, "x", 0), (0, 0, "x", 1), (1, 2, "y", 0)})
        finally:
            os.unlink(path)
        self.assertEqual(completed_keys("/no/such/file.jsonl"), set())


if __name__ == "__main__":
    unittest.main()
