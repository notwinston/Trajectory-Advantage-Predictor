"""Wave 1a engine + feature tests (CPU-only; no GPU / Prime Intellect)."""

from __future__ import annotations

import math
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

from math_loop.data import (
    GENERIC_DRIFT_PROBE_SIZE,
    GLOBAL_PROBE_SIZE,
    MATCHED_PROBE_SIZE,
    FINGERPRINT_PROBE_SIZE,
    assert_no_math500_leakage,
    build_probe_sets,
    filter_math_levels,
    parse_math_level,
    write_jsonl,
)
from math_loop import tap_probes
from math_loop import branch
from math_loop.schedule import build_tap_schedule


def _fake_math_rows():
    subjects = ["algebra", "geometry", "number_theory", "prealgebra"]
    rows = []
    for index in range(120):
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
        # there are enough algebra rows in the fake set, so matched should be all-algebra
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
        fp = tap_probes.assemble_policy_fingerprint([1.0] * 16, [0.5] * 16)
        self.assertEqual(len(fp), 32)
        self.assertEqual(fp[:16], [1.0] * 16)
        self.assertEqual(fp[16:], [0.5] * 16)


class ProbeCleanupTests(unittest.TestCase):
    def test_cuda_cleanup_releases_cache_for_cuda_device(self):
        from math_loop.probe_loss import _cleanup_torch_cuda

        calls = []

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def empty_cache():
                calls.append("empty_cache")

            @staticmethod
            def ipc_collect():
                calls.append("ipc_collect")

        class FakeTorch:
            cuda = FakeCuda()

        _cleanup_torch_cuda(FakeTorch(), "cuda")
        self.assertEqual(calls, ["empty_cache", "ipc_collect"])

    def test_cuda_cleanup_skips_cpu_device(self):
        from math_loop.probe_loss import _cleanup_torch_cuda

        calls = []

        class FakeCuda:
            @staticmethod
            def is_available():
                calls.append("is_available")
                return True

            @staticmethod
            def empty_cache():
                calls.append("empty_cache")

            @staticmethod
            def ipc_collect():
                calls.append("ipc_collect")

        class FakeTorch:
            cuda = FakeCuda()

        _cleanup_torch_cuda(FakeTorch(), "cpu")
        self.assertEqual(calls, [])


class LeakageGuardTests(unittest.TestCase):
    def test_math500_leakage_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_pool.jsonl"
            write_jsonl(path, [{"source": "math500", "split": "math500_final"}])
            with self.assertRaises(ValueError):
                assert_no_math500_leakage((path,))


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

    def test_gradient_sketch_cleans_cuda_on_return(self):
        cleanup_calls = []

        class FakeModel:
            def train(self):
                return None

        fake_torch = types.ModuleType("torch")
        fake_numpy = types.ModuleType("numpy")

        with (
            mock.patch.dict(sys.modules, {"numpy": fake_numpy, "torch": fake_torch}),
            mock.patch(
                "math_loop.probe_loss.load_model_and_tokenizer",
                return_value=(FakeModel(), object()),
            ),
            mock.patch(
                "math_loop.probe_loss._cleanup_torch_cuda",
                side_effect=lambda torch, device: cleanup_calls.append((torch, device)),
            ),
        ):
            sketch = branch.compute_lora_gradient_sketch(
                Path("/tmp/checkpoint"),
                [],
                device="cuda",
            )

        self.assertEqual(sketch, [0.0] * 64)
        self.assertEqual(cleanup_calls, [(fake_torch, "cuda")])


class ScheduleTests(unittest.TestCase):
    def test_tap_schedule_shape(self):
        prompt_ids = [f"id-{i}" for i in range(40)]
        schedule = build_tap_schedule(
            prompt_ids, chains=3, states_per_chain=8, candidates_per_state=8, prompts_per_candidate=2, seed=1729
        )
        self.assertEqual(len(schedule), 3 * 8 * 8)  # 192
        self.assertTrue(all(len(c.prompt_ids) == 2 for c in schedule))
        self.assertTrue(all(len(set(c.prompt_ids)) == 2 for c in schedule))
        # ids/shape sanity
        first = schedule[0]
        self.assertEqual(first.state_id, "0-0")
        self.assertEqual(first.candidate_id, "0-0-0")
        chains = {c.chain_index for c in schedule}
        self.assertEqual(chains, {0, 1, 2})

    def test_tap_schedule_is_deterministic(self):
        ids = [f"id-{i}" for i in range(40)]
        a = build_tap_schedule(ids, seed=1729)
        b = build_tap_schedule(ids, seed=1729)
        self.assertEqual([c.prompt_ids for c in a], [c.prompt_ids for c in b])


class TapControllerConfigTests(unittest.TestCase):
    def test_prime_rl_batch_size_is_total_samples(self):
        from argparse import Namespace

        from math_loop.tap_controller import _write_branch_config

        args = Namespace(
            prompts_per_candidate=2,
            completions_per_prompt=4,
            seq_len=4096,
            max_completion_tokens=192,
            lora_rank=16,
            learning_rate=1e-5,
            gpu_count=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _write_branch_config(
                root / "branch.toml",
                output_dir=root / "out",
                split_path=root / "split.jsonl",
                model_name="Qwen/Qwen3-8B",
                renderer="default",
                run_name="test",
                args=args,
            )
            text = cfg.read_text()
        self.assertIn("batch_size = 8", text)
        self.assertIn("group_size = 4", text)

    def test_zero_advantage_filter_is_monitor_only(self):
        from argparse import Namespace

        from math_loop.tap_controller import _write_branch_config

        args = Namespace(
            prompts_per_candidate=2,
            completions_per_prompt=4,
            seq_len=4096,
            max_completion_tokens=192,
            lora_rank=16,
            learning_rate=1e-5,
            gpu_count=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _write_branch_config(
                root / "branch.toml",
                output_dir=root / "out",
                split_path=root / "split.jsonl",
                model_name="Qwen/Qwen3-8B",
                renderer="default",
                run_name="test",
                args=args,
            )
            text = cfg.read_text()
        self.assertIn('type = "zero_advantage"\nenforce = false', text)

    def test_controller_cuda_cleanup_hook_is_deferred_and_best_effort(self):
        from math_loop.tap_controller import cleanup_cuda_if_available

        cleanup_calls = []
        fake_torch = types.ModuleType("torch")
        with (
            mock.patch.dict(sys.modules, {"torch": fake_torch}),
            mock.patch(
                "math_loop.probe_loss._cleanup_torch_cuda",
                side_effect=lambda torch, device: cleanup_calls.append((torch, device)),
            ),
        ):
            cleanup_cuda_if_available("cuda")

        self.assertEqual(cleanup_calls, [(fake_torch, "cuda")])

    def test_prune_checkpoint_weights_only_inside_output_dir(self):
        from math_loop.tap_controller import prune_checkpoint_weights

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "outputs"
            checkpoint = output_dir / "tap" / "run" / "branches" / "b0" / "weights" / "step_1"
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.safetensors").write_text("large", encoding="utf-8")
            self.assertTrue(prune_checkpoint_weights(checkpoint, output_dir))
            self.assertFalse((checkpoint.parent).exists())

            outside = root / "outside" / "weights" / "step_1"
            outside.mkdir(parents=True)
            (outside / "model.safetensors").write_text("keep", encoding="utf-8")
            self.assertFalse(prune_checkpoint_weights(outside, output_dir))
            self.assertTrue(outside.exists())

    def test_prune_prime_rl_checkpoints_only_inside_output_dir(self):
        from math_loop.tap_controller import prune_prime_rl_checkpoints

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "outputs"
            branch_out = output_dir / "tap" / "run" / "branches" / "state_0-0" / "cand_0"
            checkpoint_dir = branch_out / "checkpoints" / "step_1"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "trainer.pt").write_text("large", encoding="utf-8")
            weights_dir = branch_out / "weights" / "step_1"
            weights_dir.mkdir(parents=True)
            (weights_dir / "model.safetensors").write_text("keep", encoding="utf-8")

            self.assertTrue(prune_prime_rl_checkpoints(branch_out, output_dir))
            self.assertFalse((branch_out / "checkpoints").exists())
            self.assertTrue(weights_dir.exists())

            outside = root / "outside"
            (outside / "checkpoints").mkdir(parents=True)
            self.assertFalse(prune_prime_rl_checkpoints(outside, output_dir))
            self.assertTrue((outside / "checkpoints").exists())

    def test_output_download_excludes_weight_and_checkpoint_trees(self):
        from run_prime_rl_math_loop import download_outputs

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch("run_prime_rl_math_loop.shutil.which", return_value="/usr/bin/rsync"),
            mock.patch("run_prime_rl_math_loop.run") as run_mock,
        ):
            download_outputs(["ssh"], "root@example", Path(tmp))
        command = run_mock.call_args.args[0]
        self.assertIn("--exclude", command)
        self.assertIn("*/weights/**", command)
        self.assertIn("*/checkpoints/**", command)

    def test_rollout_sequence_length_falls_back_when_prime_omits_tokens(self):
        from math_loop.tap_controller import _map_rollout_row

        row = {
            "prompt": "What is 1+1?",
            "completion": "Reason briefly. \\boxed{2}",
            "reward": 1.0,
        }

        mapped = _map_rollout_row(row, 0, "0-0-0")
        self.assertGreater(mapped["sequence_length"], 0)


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
                expected = 1000.0 * (0.75 * matched_gain + 0.25 * global_gain - 0.05 * max(incr, 0.0))
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


import importlib.util as _ilu

_HAS_TORCH = _ilu.find_spec("torch") is not None


@unittest.skipUnless(_HAS_TORCH, "torch required for vectorized-probe equivalence")
class VectorizedProbeMathTests(unittest.TestCase):
    """The torch-vectorized KL/NLL/entropy must equal the pure-Python aggregators
    (fp32 tolerance), since they feed identical utility_points labels."""

    def _logits(self, rows, cols, seed):
        import torch

        return torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed))

    def test_kl_matches_sequence_kl(self):
        base, branch_l = self._logits(7, 19, 1), self._logits(7, 19, 2)
        self.assertAlmostEqual(
            tap_probes.kl_mean_vectorized(base, branch_l),
            tap_probes.sequence_kl(base.tolist(), branch_l.tolist()),
            places=5,
        )

    def test_nll_matches_pure(self):
        import random

        base = self._logits(6, 13, 3)
        targets = [random.Random(0).randrange(13) for _ in range(6)]
        self.assertAlmostEqual(
            tap_probes.nll_mean_vectorized(base, targets),
            tap_probes.nll_from_logits_and_targets(base.tolist(), targets),
            places=5,
        )

    def test_entropy_matches_pure(self):
        base = self._logits(5, 11, 4)
        pure = sum(tap_probes.sequence_entropy(base.tolist())) / 5
        self.assertAlmostEqual(tap_probes.entropy_mean_vectorized(base), pure, places=5)

    def test_nll_empty_and_out_of_range(self):
        import torch

        self.assertEqual(tap_probes.nll_mean_vectorized(torch.zeros(0, 5), []), 0.0)
        base = self._logits(4, 7, 5)
        oob = [99, 99, 99, 99]  # all out of vocab -> pure-Python skips them -> 0.0
        self.assertAlmostEqual(
            tap_probes.nll_mean_vectorized(base, oob),
            tap_probes.nll_from_logits_and_targets(base.tolist(), oob),
            places=5,
        )


class ProbeSessionTests(unittest.TestCase):
    def test_constructs_without_heavy_imports(self):
        from math_loop.probe_loss import ProbeSession

        with mock.patch.dict(sys.modules, {"torch": None, "transformers": None, "peft": None}):
            session = ProbeSession(device="cpu")
            self.assertEqual(session.model_name, "Qwen/Qwen3-8B")
            self.assertIsNone(session._base)

    def test_grad_sketch_zeroed_under_tap_no_torch(self):
        import os

        from math_loop.probe_loss import ProbeSession

        prior = os.environ.get("TAP_NO_TORCH")
        os.environ["TAP_NO_TORCH"] = "1"
        try:
            self.assertEqual(ProbeSession(device="cpu").grad_sketch([{"prompt_text": "x"}]), [0.0] * 64)
        finally:
            if prior is None:
                os.environ.pop("TAP_NO_TORCH", None)
            else:
                os.environ["TAP_NO_TORCH"] = prior

    def test_use_adapter_loads_once_then_set_adapter(self):
        from math_loop.probe_loss import ProbeSession

        calls = {"load": [], "set": []}

        class FakePeft:
            def load_adapter(self, path, adapter_name):
                calls["load"].append(adapter_name)

            def set_adapter(self, name):
                calls["set"].append(name)

        session = ProbeSession(device="cpu")
        session._base = object()      # pretend the base model is already resident
        session._model = FakePeft()   # ... wrapped in a PeftModel
        with mock.patch("math_loop.probe_loss.find_adapter_path", return_value=Path("/ckpt/lora_adapters")):
            n1 = session.use_adapter(Path("/ckpt"))
            n2 = session.use_adapter(Path("/ckpt"))  # same path -> short-circuit
        self.assertEqual(n1, n2)
        self.assertEqual(len(calls["load"]), 1)     # adapter loaded exactly once
        self.assertEqual(calls["set"], [n1, n1])    # set_adapter on both calls


class PersistentInferenceConfigTests(unittest.TestCase):
    def _spec(self, **kw):
        from math_loop.prime_rl_config import PrimeRLConfigSpec

        base = dict(
            output_dir=Path("/o"), split_path=Path("/s.jsonl"), max_steps=1,
            batch_size=8, group_size=4, model_name="Qwen/Qwen3-8B", lora_rank=16,
        )
        base.update(kw)
        return PrimeRLConfigSpec(**base)

    def test_inference_server_config_is_lora_and_eager(self):
        from math_loop.prime_rl_config import render_inference_server_config

        text = render_inference_server_config(self._spec())
        self.assertIn("enable_lora = true", text)
        self.assertIn("enforce_eager = true", text)
        self.assertIn("max_lora_rank = 16", text)
        self.assertIn('name = "Qwen/Qwen3-8B"', text)

    def test_persistent_branch_omits_inference_block(self):
        from math_loop.prime_rl_config import render_prime_rl_config

        self.assertNotIn("[inference]", render_prime_rl_config(self._spec(include_inference=False)))

    def test_self_contained_config_keeps_inference_with_eager(self):
        from math_loop.prime_rl_config import render_prime_rl_config

        text = render_prime_rl_config(self._spec(include_inference=True))
        self.assertIn("[inference]", text)
        self.assertIn("enforce_eager = true", text)


class BackgroundProcessTests(unittest.TestCase):
    def test_wait_returns_false_when_process_dies(self):
        from math_loop.controller import wait_for_http_ready

        class DeadProc:
            def poll(self):
                return 1  # already exited

        self.assertFalse(
            wait_for_http_ready("http://127.0.0.1:9/health", timeout=1.0, interval=0.01, proc=DeadProc())
        )

    def test_stop_process_is_idempotent(self):
        from math_loop.controller import stop_process

        class ExitedProc:
            def poll(self):
                return 0

            def terminate(self):  # pragma: no cover - must not be called
                raise AssertionError("must not terminate an already-exited process")

        stop_process(ExitedProc())  # no-op, no exception
        stop_process(None)          # tolerates None


if __name__ == "__main__":
    unittest.main()
