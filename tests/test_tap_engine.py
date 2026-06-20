"""Wave 1a engine + feature tests (CPU-only; no GPU / Prime Intellect)."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from math_loop.data import (
    GENERIC_DRIFT_PROBE_SIZE,
    GLOBAL_PROBE_SIZE,
    MATCHED_PROBE_SIZE,
    FINGERPRINT_PROBE_SIZE,
    build_probe_sets,
    filter_math_levels,
    parse_math_level,
)
from math_loop import tap_probes
from math_loop import branch
from math_loop.schedule import build_tap_schedule


def _fake_math_rows():
    subjects = ["algebra", "geometry", "number_theory", "prealgebra"]
    rows = []
    for index in range(60):
        level = (index % 6) + 1  # levels 1..6
        rows.append(
            {
                "problem": f"Problem {index}: compute {index} + {index}.",
                "solution": f"The answer is \\boxed{{{index * 2}}}.",
                "level": f"Level {level}",
                "type": subjects[index % len(subjects)],
            }
        )
    return rows


class DataLevelTests(unittest.TestCase):
    def test_parse_math_level(self):
        self.assertEqual(parse_math_level("Level 3"), 3)
        self.assertEqual(parse_math_level("Level 5"), 5)
        self.assertIsNone(parse_math_level("Level ?"))
        self.assertIsNone(parse_math_level(None))

    def test_filter_keeps_only_levels_3_to_5(self):
        kept = filter_math_levels(_fake_math_rows())
        self.assertTrue(kept, "expected some level 3-5 rows")
        self.assertTrue(all(row["level"] in (3, 4, 5) for row in kept))
        self.assertTrue(all("subject" in row for row in kept))

    def test_filter_drops_unparseable_level(self):
        rows = [{"problem": "p", "solution": "\\boxed{1}", "level": "Level ?"}]
        self.assertEqual(filter_math_levels(rows), [])


class ProbeSetTests(unittest.TestCase):
    def setUp(self):
        self.probe_rows = filter_math_levels(_fake_math_rows())

    def test_probe_sizes(self):
        sets = build_probe_sets(self.probe_rows, subject="algebra", seed=1729)
        self.assertEqual(len(sets.matched), MATCHED_PROBE_SIZE)
        self.assertEqual(len(sets.global_probe), GLOBAL_PROBE_SIZE)
        self.assertEqual(len(sets.generic_drift), GENERIC_DRIFT_PROBE_SIZE)
        self.assertEqual(len(sets.fingerprint), FINGERPRINT_PROBE_SIZE)

    def test_determinism_at_fixed_seed(self):
        a = build_probe_sets(self.probe_rows, subject="geometry", seed=7)
        b = build_probe_sets(self.probe_rows, subject="geometry", seed=7)
        self.assertEqual([r["id"] for r in a.matched], [r["id"] for r in b.matched])
        self.assertEqual([r["id"] for r in a.global_probe], [r["id"] for r in b.global_probe])
        self.assertEqual([r["id"] for r in a.fingerprint], [r["id"] for r in b.fingerprint])

    def test_matched_prefers_requested_subject(self):
        sets = build_probe_sets(self.probe_rows, subject="algebra", seed=1729)
        algebra = [r for r in sets.matched if r.get("subject") == "algebra"]
        # there are >=8 algebra rows in the fake set, so matched should be all-algebra
        self.assertEqual(len(algebra), MATCHED_PROBE_SIZE)

    def test_global_probe_is_stratified(self):
        sets = build_probe_sets(self.probe_rows, seed=3)
        levels = {r["level"] for r in sets.global_probe}
        self.assertTrue(levels.issubset({3, 4, 5}))
        self.assertGreaterEqual(len(levels), 2)

    def test_fingerprint_is_fixed_across_subjects(self):
        a = build_probe_sets(self.probe_rows, subject="algebra", seed=1)
        b = build_probe_sets(self.probe_rows, subject="geometry", seed=99)
        self.assertEqual([r["id"] for r in a.fingerprint], [r["id"] for r in b.fingerprint])


class ProbeMathTests(unittest.TestCase):
    def test_entropy_of_uniform_logits(self):
        logits = [0.0, 0.0, 0.0, 0.0]  # uniform over 4 -> entropy ln(4)
        self.assertAlmostEqual(tap_probes.entropy_from_logits(logits), math.log(4), places=6)

    def test_nll_from_logprobs(self):
        self.assertAlmostEqual(tap_probes.token_nll_from_logprobs([-1.0, -2.0, -3.0]), 2.0, places=9)

    def test_nll_from_logits_and_targets_uniform(self):
        logits = [[0.0, 0.0, 0.0, 0.0]] * 3
        self.assertAlmostEqual(
            tap_probes.nll_from_logits_and_targets(logits, [0, 1, 2]), math.log(4), places=6
        )

    def test_kl_self_is_zero(self):
        logits = [[1.0, 2.0, 0.5]] * 4
        self.assertAlmostEqual(tap_probes.sequence_kl(logits, logits), 0.0, places=9)

    def test_kl_is_positive_for_different_dists(self):
        base = [[2.0, 0.0, 0.0]]
        branch_logits = [[0.0, 0.0, 2.0]]
        self.assertGreater(tap_probes.sequence_kl(base, branch_logits), 0.0)

    def test_fingerprint_is_16_values(self):
        fp = tap_probes.assemble_policy_fingerprint([1.0] * 8, [0.5] * 8)
        self.assertEqual(len(fp), 16)
        self.assertEqual(fp[:8], [1.0] * 8)
        self.assertEqual(fp[8:], [0.5] * 8)


class BranchPrimitiveTests(unittest.TestCase):
    def test_branch_command_contains_resume_step(self):
        cmd = branch.branch_command("uv run rl", "configs/x.toml", resume_step=3)
        self.assertIn("--ckpt.resume-step", cmd)
        self.assertIn("3", cmd)
        self.assertIn("@", cmd)

    def test_identical_before_state_hashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "cand_0" / "before"
            b = root / "cand_1" / "before"
            a.mkdir(parents=True)
            b.mkdir(parents=True)
            (a / "adapter_model.bin").write_bytes(b"identical-weights")
            (b / "adapter_model.bin").write_bytes(b"identical-weights")
            shared = branch.assert_identical_before_state([a, b])
            self.assertIn("checkpoint_hash", shared)
            self.assertEqual(
                branch.read_before_state_hashes(a)["checkpoint_hash"],
                branch.read_before_state_hashes(b)["checkpoint_hash"],
            )

    def test_mismatched_before_state_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "cand_0" / "before"
            b = root / "cand_1" / "before"
            a.mkdir(parents=True)
            b.mkdir(parents=True)
            (a / "adapter_model.bin").write_bytes(b"weights-A")
            (b / "adapter_model.bin").write_bytes(b"weights-B-different")
            with self.assertRaises(ValueError):
                branch.assert_identical_before_state([a, b])

    def test_artifact_layout_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand_dir = branch.candidate_dir(tmp, "0-2", 5)
            self.assertTrue(str(cand_dir).endswith("state_0-2/cand_5"))
            branch.write_candidate_artifacts(
                cand_dir,
                rollouts=[{"trajectory_id": "t0"}],
                probe_before={"matched_probe_nll": 1.0},
                probe_after={"matched_probe_nll": 0.9},
                grad_sketch=None,
            )
            self.assertTrue((cand_dir / "rollouts.jsonl").exists())
            self.assertTrue((cand_dir / "probe_before.json").exists())
            self.assertTrue((cand_dir / "probe_after.json").exists())
            self.assertTrue((cand_dir / "grad_unavailable.flag").exists())


class ScheduleTests(unittest.TestCase):
    def test_tap_schedule_shape(self):
        prompt_ids = [f"id-{i}" for i in range(40)]
        schedule = build_tap_schedule(
            prompt_ids, chains=2, states_per_chain=6, candidates_per_state=6, prompts_per_candidate=2, seed=1729
        )
        self.assertEqual(len(schedule), 2 * 6 * 6)  # 72
        self.assertTrue(all(len(c.prompt_ids) == 2 for c in schedule))
        self.assertTrue(all(len(set(c.prompt_ids)) == 2 for c in schedule))
        # ids/shape sanity
        first = schedule[0]
        self.assertEqual(first.state_id, "0-0")
        self.assertEqual(first.candidate_id, "0-0-0")
        chains = {c.chain_index for c in schedule}
        self.assertEqual(chains, {0, 1})

    def test_tap_schedule_is_deterministic(self):
        ids = [f"id-{i}" for i in range(40)]
        a = build_tap_schedule(ids, seed=1729)
        b = build_tap_schedule(ids, seed=1729)
        self.assertEqual([c.prompt_ids for c in a], [c.prompt_ids for c in b])


class FeatureExtractorTests(unittest.TestCase):
    FIXTURE = Path("tests/fixtures/raw_artifacts")

    def _convert(self, out_dir: Path):
        from math_loop import features

        return features.convert(self.FIXTURE, out_dir)

    def test_fixture_converts_to_valid_parquet(self):
        from tap.schema import validate_parquet_dir

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            counts = self._convert(out)
            self.assertEqual(counts["states.parquet"], 1)
            self.assertEqual(counts["candidates.parquet"], 2)
            self.assertEqual(counts["trajectories.parquet"], 16)  # 2 cands x 8
            # 4 files exist and pass the frozen schema contract.
            validate_parquet_dir(out)

    def test_gradient_sketch_paths_and_no_nan(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._convert(out)
            table = pq.read_table(out / "candidates.parquet").to_pylist()
            by_id = {row["candidate_id"]: row for row in table}
            cand0 = by_id["0-0-0"]["gradient_sketch"]
            cand1 = by_id["0-0-1"]["gradient_sketch"]
            self.assertEqual(len(cand0), 64)
            self.assertEqual(len(cand1), 64)
            # cand0 has a real sketch (nonzero); cand1 fell back to zeros.
            self.assertTrue(any(abs(v) > 0 for v in cand0))
            self.assertTrue(all(v == 0.0 for v in cand1))
            # no NaN anywhere in either sketch
            for sketch in (cand0, cand1):
                self.assertFalse(any(math.isnan(v) for v in sketch))

    def test_utility_recomputed_via_join(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._convert(out)
            states = {r["state_id"]: r for r in pq.read_table(out / "states.parquet").to_pylist()}
            candidates = pq.read_table(out / "candidates.parquet").to_pylist()
            for cand in candidates:
                state = states[cand["state_id"]]
                matched_gain = state["matched_probe_nll_before"] - cand["matched_probe_nll_after"]
                global_gain = state["global_probe_nll_before"] - cand["global_probe_nll_after"]
                incr = cand["generic_kl_after"] - state["generic_kl_before"]
                expected = 1000.0 * (0.8 * matched_gain + 0.2 * global_gain - 0.03 * max(incr, 0.0))
                self.assertAlmostEqual(cand["utility_points"], expected, delta=1e-6)
                self.assertAlmostEqual(cand["matched_gain"], matched_gain, delta=1e-6)

    def test_selected_candidate_flag(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._convert(out)
            candidates = {r["candidate_id"]: r for r in pq.read_table(out / "candidates.parquet").to_pylist()}
            self.assertTrue(candidates["0-0-0"]["is_selected_for_main_chain"])
            self.assertFalse(candidates["0-0-1"]["is_selected_for_main_chain"])

    def test_no_nan_in_gradient_sketch_under_tap_no_torch(self):
        import os
        import pyarrow.parquet as pq

        prior = os.environ.get("TAP_NO_TORCH")
        os.environ["TAP_NO_TORCH"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                self._convert(out)
                table = pq.read_table(out / "candidates.parquet").to_pylist()
                for row in table:
                    sketch = row["gradient_sketch"]
                    self.assertEqual(len(sketch), 64)
                    self.assertTrue(all(v == 0.0 for v in sketch))  # zeroed under TAP_NO_TORCH
        finally:
            if prior is None:
                os.environ.pop("TAP_NO_TORCH", None)
            else:
                os.environ["TAP_NO_TORCH"] = prior


if __name__ == "__main__":
    unittest.main()
