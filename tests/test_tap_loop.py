import json
from pathlib import Path
import contextlib
import io
import tempfile
import unittest

from tap_loop.artifacts import TapArtifactWriter
from tap_loop.data import (
    assert_no_math500_leakage,
    normalize_tap_math_rows,
    split_tap_math_rows,
    write_jsonl,
)
from tap_loop.metrics import evaluate_ranker, pairwise_accuracy, spearman, top_one_regret
from tap_loop.probes import select_global_probe, select_matched_probe, utility_points
from tap_loop.schedule import build_tap_candidate_schedule, latest_history, select_main_candidate
from tap_loop.training import pairwise_ranking_loss, train_ridge
from tap_loop.train_tap import run_training
from tap_loop.collector import CollectorConfig, DryRunPolicyBackend, run_collection
from run_tap_prime import build_bootstrap_command, parse_args


def math_row(index, *, level=3, subject="Algebra"):
    return {
        "problem": f"Problem {index}",
        "solution": f"Solution \\boxed{{{index}}}",
        "level": f"Level {level}",
        "type": subject,
    }


class TapDataTests(unittest.TestCase):
    def test_filters_to_levels_three_through_five_and_requires_metadata(self):
        rows = [math_row(1, level=2), math_row(2, level=3), math_row(3, level=5)]
        normalized = normalize_tap_math_rows(rows)
        self.assertEqual([row["level"] for row in normalized], [3, 5])
        with self.assertRaisesRegex(ValueError, "missing required TAP fields"):
            normalize_tap_math_rows([{"problem": "p", "solution": "\\boxed{1}"}])

    def test_split_is_deterministic_and_math500_guard_catches_leakage(self):
        rows = [math_row(index, level=3 + index % 3, subject="Algebra") for index in range(40)]
        train_a, heldout_a = split_tap_math_rows(rows, heldout_size=8, seed=7)
        train_b, heldout_b = split_tap_math_rows(rows, heldout_size=8, seed=7)
        self.assertEqual([row["id"] for row in heldout_a], [row["id"] for row in heldout_b])
        self.assertEqual(len(heldout_a), 8)
        self.assertTrue({row["id"] for row in train_a}.isdisjoint({row["id"] for row in heldout_a}))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            write_jsonl(path, [{"source": "math500", "split": "math500_final"}])
            with self.assertRaisesRegex(ValueError, "MATH-500"):
                assert_no_math500_leakage([path])


class TapScheduleTests(unittest.TestCase):
    def test_candidate_schedule_and_selection_are_deterministic(self):
        schedule_a = build_tap_candidate_schedule(["a", "b", "c"], chains=2, states_per_chain=2, candidates_per_state=3)
        schedule_b = build_tap_candidate_schedule(["a", "b", "c"], chains=2, states_per_chain=2, candidates_per_state=3)
        self.assertEqual(schedule_a, schedule_b)
        self.assertEqual(len(schedule_a), 12)
        self.assertEqual(schedule_a[0].state_id, "chain_00_state_000")
        self.assertEqual(schedule_a[0].candidate_id, "chain_00_state_000_candidate_00")
        self.assertEqual(
            select_main_candidate(6, chain_id=1, state_index=3, seed=9),
            select_main_candidate(6, chain_id=1, state_index=3, seed=9),
        )
        self.assertEqual(latest_history(["a", "b", "c", "d", "e"]), ["b", "c", "d", "e"])


class TapProbeTests(unittest.TestCase):
    def test_probe_selection_and_utility_math(self):
        heldout = []
        for index in range(30):
            subject = "Algebra" if index % 2 == 0 else "Geometry"
            heldout.append({**math_row(index, level=3 + index % 3, subject=subject), "id": f"h-{index}", "question": f"Q{index}", "answer": str(index)})
            heldout[-1]["subject"] = subject
            heldout[-1]["level"] = 3 + index % 3
        candidate = [
            {"id": "c-a", "subject": "Algebra", "level": 4, "question": "qa"},
            {"id": "c-g", "subject": "Geometry", "level": 5, "question": "qg"},
        ]
        matched = select_matched_probe(candidate, heldout, size=8, seed=3)
        global_probe = select_global_probe(heldout, size=8, seed=3)
        self.assertEqual(len(matched), 8)
        self.assertEqual(len(global_probe), 8)
        self.assertEqual(len([row for row in matched if row["subject"] == "Algebra"]), 4)
        result = utility_points(2.0, 1.9, 2.5, 2.4, 0.1, 0.2)
        self.assertAlmostEqual(result["matched_gain"], 0.1)
        self.assertAlmostEqual(result["global_gain"], 0.1)
        self.assertAlmostEqual(result["incremental_generic_kl"], 0.1)
        self.assertAlmostEqual(result["utility_points"], 97.0)


class TapArtifactTests(unittest.TestCase):
    def test_fragment_writer_validates_vectors_and_compacts_without_pyarrow(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = TapArtifactWriter(Path(tmp), require_parquet=False)
            row = {"state_id": "s", "policy_fingerprint": [0.0] * 16, "value": 1.0}
            writer.write_fragment("states", [row], fragment_id="s")
            compacted = writer.compact_table("states")
            self.assertTrue(compacted.exists())
            self.assertIn(compacted.suffix, {".jsonl", ".parquet"})
            with self.assertRaisesRegex(ValueError, "policy_fingerprint"):
                writer.write_fragment("states", [{"policy_fingerprint": [0.0]}], fragment_id="bad")


class TapMetricTrainingTests(unittest.TestCase):
    def test_ranking_metrics_and_ridge_baseline(self):
        self.assertAlmostEqual(spearman([1, 2, 3], [1, 4, 9]), 1.0)
        self.assertAlmostEqual(pairwise_accuracy([1, 2, 3], [1, 0, 3]), 2 / 3)
        self.assertEqual(top_one_regret([0, 1], [5, 3]), 2.0)
        rows = []
        for chain in (0, 1):
            for state in range(2):
                for candidate in range(3):
                    rows.append(
                        {
                            "state_id": f"c{chain}s{state}",
                            "chain_id": chain,
                            "candidate_index": candidate,
                            "candidate_reward_mean": float(candidate),
                            "utility_points": float(candidate * 2 + chain),
                        }
                    )
        report = evaluate_ranker(rows, lambda row: row["candidate_reward_mean"])
        self.assertEqual(report["states"], 4.0)
        self.assertAlmostEqual(report["pairwise_accuracy"], 1.0)
        model = train_ridge(rows)
        self.assertEqual(len(model.predict(rows)), len(rows))
        self.assertGreater(pairwise_ranking_loss([0.0, 1.0], [1.0, 0.0]), 0.0)

    def test_training_cli_writes_metrics_report_from_jsonl_compaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for chain in (0, 1):
                for state in range(2):
                    for candidate in range(3):
                        rows.append(
                            {
                                "state_id": f"c{chain}s{state}",
                                "chain_id": chain,
                                "candidate_id": f"c{chain}s{state}k{candidate}",
                                "candidate_index": candidate,
                                "candidate_reward_mean": float(candidate),
                                "candidate_advantage_mean": float(candidate),
                                "candidate_geometric_mean_probability": 0.1 * candidate,
                                "candidate_arithmetic_mean_probability": 0.1 * candidate,
                                "candidate_mean_log_probability": -float(candidate + 1),
                                "gradient_norm": float(candidate),
                                "matched_probe_gradient_alignment": float(candidate),
                                "max_semantic_similarity_to_history": 0.0,
                                "candidate_embedding": [0.0] * 256,
                                "gradient_sketch": [0.0] * 64,
                                "utility_points": float(candidate * 2 + chain),
                            }
                        )
            write_jsonl(root / "parquet" / "candidates.jsonl", rows)
            result = run_training(root)
            self.assertEqual(result["candidate_rows"], len(rows))
            self.assertIn("ridge", result["baselines"])
            self.assertTrue((root / "reports" / "tap_metrics.json").exists())

    def test_training_allows_one_chain_smoke_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for state in range(2):
                for candidate in range(3):
                    rows.append(
                        {
                            "state_id": f"s{state}",
                            "chain_id": 0,
                            "candidate_id": f"s{state}k{candidate}",
                            "candidate_index": candidate,
                            "candidate_reward_mean": float(candidate),
                            "candidate_embedding": [0.0] * 256,
                            "gradient_sketch": [0.0] * 64,
                            "utility_points": float(candidate),
                        }
                    )
            write_jsonl(root / "parquet" / "candidates.jsonl", rows)
            result = run_training(root)
            self.assertNotIn("ridge", result["baselines"])
            self.assertEqual(result["candidate_rows"], len(rows))


class TapCollectorTests(unittest.TestCase):
    def test_dry_collection_writes_pre_branch_state_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = []
            heldout = []
            for index in range(20):
                row = {
                    "id": f"p-{index}",
                    "question": f"Question {index}",
                    "problem": f"Question {index}",
                    "solution": f"Solution \\boxed{{{index}}}",
                    "answer": str(index),
                    "subject": "Algebra" if index % 2 else "Geometry",
                    "difficulty": 3 + index % 3,
                    "level": 3 + index % 3,
                    "split": "tap_train_pool",
                }
                (train if index < 10 else heldout).append(row)
            data_dir = root / "data"
            write_jsonl(data_dir / "tap_math_l3_5_train_pool.jsonl", train)
            write_jsonl(data_dir / "tap_math_l3_5_heldout256.jsonl", heldout)
            write_jsonl(data_dir / "generic_drift_prompts.jsonl", [{"id": "g", "prompt": "hello"}])
            result = run_collection(
                CollectorConfig(root, chains=1, states_per_chain=1, candidates_per_state=2),
                DryRunPolicyBackend(root),
            )
            self.assertEqual(result["completed_candidates"], 2)
            resume_result = run_collection(
                CollectorConfig(root, chains=1, states_per_chain=1, candidates_per_state=2),
                DryRunPolicyBackend(root),
            )
            self.assertEqual(resume_result["completed_candidates"], 0)
            states = [json.loads(line) for line in (root / "parquet" / "states.jsonl").read_text().splitlines() if line]
            self.assertEqual(states[0]["history_candidate_ids"], [])
            self.assertTrue((root / "fragments" / "history" / "chain_00_state_000.jsonl").exists())
            self.assertTrue((root / "checkpoints" / "chains" / "chain_00" / "state_001" / "state.json").exists())
            status = json.loads((root / "collection_status.json").read_text(encoding="utf-8"))
            self.assertIn(status["status"], {"complete", "resumed_skip_completed_state"})


class TapLauncherTests(unittest.TestCase):
    def test_launcher_dry_run_writes_manifest_without_prime_cli(self):
        from run_tap_prime import main

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "tap"
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "--dry-run",
                        "--ephemeral-ok",
                        "--backend",
                        "dry-run",
                        "--run-id",
                        "tap_test",
                        "--output-dir",
                        str(output),
                        "--ssh-key",
                        str(Path(tmp) / "missing_key"),
                    ]
                )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_id"], "tap_test")
            self.assertEqual(manifest["backend"], "dry-run")

    def test_bootstrap_uses_hf_token_env_without_printing_secret(self):
        args = parse_args(["--dry-run", "--ephemeral-ok", "--hf-token", "secret-token"])
        command = build_bootstrap_command(args, Path("/workspace/tap_runs/example"))
        rendered = " ".join(command)
        self.assertIn("HF_TOKEN", rendered)
        self.assertNotIn("secret-token", rendered)


if __name__ == "__main__":
    unittest.main()
