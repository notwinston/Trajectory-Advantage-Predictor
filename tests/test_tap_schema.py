"""Tests for the TAP v1 frozen schema contract and synthetic generator."""

import math
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from tap import schema
from tap.synth import generate, resolve_layout


def _read(directory: Path, name: str):
    return pq.read_table(directory / name).to_pylist()


class ResolveLayoutTests(unittest.TestCase):
    def test_canonical_layouts(self):
        self.assertEqual(resolve_layout(48, 2, 6), (4, 6))
        self.assertEqual(resolve_layout(72, 2, 6), (6, 6))
        # 128 with default cps=6 is indivisible -> canonical (8, 8).
        self.assertEqual(resolve_layout(128, 2, 6), (8, 8))
        self.assertEqual(resolve_layout(128, 2, 8), (8, 8))

    def test_indivisible_raises(self):
        with self.assertRaises(ValueError):
            resolve_layout(50, 2, 6)


class SchemaContractTests(unittest.TestCase):
    def test_column_order_matches_spec_per_file(self):
        for name in schema.FILE_NAMES:
            cols = schema.column_names(name)
            self.assertEqual(len(cols), len(set(cols)), f"{name} has duplicate columns")
        # spec uses estimated_update_norm (not update_norm) in candidates.
        self.assertIn("estimated_update_norm", schema.column_names("candidates.parquet"))
        self.assertNotIn("update_norm", schema.column_names("candidates.parquet"))

    def test_frozen_vector_widths(self):
        self.assertEqual(schema.POLICY_FINGERPRINT_DIM, 16)
        self.assertEqual(schema.CANDIDATE_EMBEDDING_DIM, 256)
        self.assertEqual(schema.GRADIENT_SKETCH_DIM, 64)
        self.assertEqual(schema.TRAJECTORY_EMBEDDING_DIM, 128)

    def test_validate_rejects_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            generate(tmp, labels=48, chains=2, candidates_per_state=6, seed=1729)
            (Path(tmp) / "history.parquet").unlink()
            with self.assertRaises(FileNotFoundError):
                schema.validate_parquet_dir(tmp)


class SynthGenerationTests(unittest.TestCase):
    def _generate(self, tmp, labels):
        counts = generate(tmp, labels=labels, chains=2, candidates_per_state=6, seed=1729)
        schema.validate_parquet_dir(tmp)
        return counts

    def test_each_label_count_is_schema_valid(self):
        layouts = {48: (4, 6), 72: (6, 6), 128: (8, 8)}
        for labels, (states_per_chain, cps) in layouts.items():
            with self.subTest(labels=labels):
                with tempfile.TemporaryDirectory() as tmp:
                    counts = self._generate(tmp, labels)
                    self.assertEqual(counts["candidates.parquet"], labels)
                    self.assertEqual(counts["states.parquet"], 2 * states_per_chain)
                    self.assertEqual(counts["trajectories.parquet"], labels * 8)
                    # history: per chain sum_{i} min(i, 4) applied updates.
                    per_chain_history = sum(min(i, 4) for i in range(states_per_chain))
                    self.assertEqual(counts["history.parquet"], 2 * per_chain_history)

    def test_vector_widths_in_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._generate(tmp, 72)
            states = _read(Path(tmp), "states.parquet")
            candidates = _read(Path(tmp), "candidates.parquet")
            trajectories = _read(Path(tmp), "trajectories.parquet")
            self.assertTrue(all(len(r["policy_fingerprint"]) == 16 for r in states))
            self.assertTrue(all(len(r["candidate_embedding"]) == 256 for r in candidates))
            self.assertTrue(all(len(r["gradient_sketch"]) == 64 for r in candidates))
            self.assertTrue(all(len(r["trajectory_embedding"]) == 128 for r in trajectories))

    def test_no_nan_in_gradient_sketch(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._generate(tmp, 72)
            candidates = _read(Path(tmp), "candidates.parquet")
            for row in candidates:
                self.assertFalse(any(math.isnan(v) for v in row["gradient_sketch"]))

    def test_utility_recompute_to_1e_6(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._generate(tmp, 72)
            states = {r["state_id"]: r for r in _read(Path(tmp), "states.parquet")}
            candidates = _read(Path(tmp), "candidates.parquet")
            for cand in candidates:
                state = states[cand["state_id"]]
                matched_gain = state["matched_probe_nll_before"] - cand["matched_probe_nll_after"]
                global_gain = state["global_probe_nll_before"] - cand["global_probe_nll_after"]
                incremental = cand["generic_kl_after"] - state["generic_kl_before"]
                expected = 1000.0 * (
                    0.8 * matched_gain
                    + 0.2 * global_gain
                    - 0.03 * max(incremental, 0.0)
                )
                self.assertAlmostEqual(cand["utility_points"], expected, delta=1e-6)

    def test_exactly_one_selected_per_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._generate(tmp, 72)
            candidates = _read(Path(tmp), "candidates.parquet")
            by_state = {}
            for cand in candidates:
                by_state.setdefault(cand["state_id"], []).append(cand["is_selected_for_main_chain"])
            for state_id, flags in by_state.items():
                self.assertEqual(sum(bool(f) for f in flags), 1, f"state {state_id}")

    def test_candidate_aggregates_match_trajectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._generate(tmp, 48)
            trajectories = _read(Path(tmp), "trajectories.parquet")
            candidates = _read(Path(tmp), "candidates.parquet")
            by_candidate = {}
            for traj in trajectories:
                by_candidate.setdefault(traj["candidate_id"], []).append(traj)
            for cand in candidates:
                trajs = by_candidate[cand["candidate_id"]]
                self.assertEqual(len(trajs), 8)
                reward_mean = sum(t["reward_total"] for t in trajs) / len(trajs)
                self.assertAlmostEqual(cand["candidate_reward_mean"], reward_mean, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
