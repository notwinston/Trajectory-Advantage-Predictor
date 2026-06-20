import unittest
from pathlib import Path
from types import SimpleNamespace
import tempfile

from math_loop.answers import (
    NON_THINKING_SYSTEM_PROMPT,
    exact_match,
    extract_boxed_answer,
)
from math_loop.controller import checkpoint_steps, rl_command
from math_loop.data import normalize_math500_rows, split_math_rows
from math_loop.prime_rl_config import PrimeRLConfigSpec, render_prime_rl_config
from math_loop.schedule import build_candidate_schedule


class AnswerTests(unittest.TestCase):
    def test_extracts_last_nested_boxed_answer(self):
        text = "first \\boxed{1} then \\boxed{\\frac{2}{3}}"
        self.assertEqual(extract_boxed_answer(text, strict=True), "\\frac{2}{3}")

    def test_exact_match_normalizes_spacing_and_box(self):
        self.assertTrue(exact_match("\\boxed{ 4 }", "4"))
        self.assertTrue(exact_match("\\left( x+1 \\right)", "(x+1)"))

    def test_non_thinking_prompt_has_no_think(self):
        self.assertIn("/no_think", NON_THINKING_SYSTEM_PROMPT)


class DataTests(unittest.TestCase):
    def test_training_split_is_deterministic_and_disjoint(self):
        rows = [
            {"problem": f"Problem {index}", "solution": f"Solution \\boxed{{{index}}}"}
            for index in range(140)
        ]
        train_a, probe_a = split_math_rows(rows, probe_size=128, seed=7)
        train_b, probe_b = split_math_rows(rows, probe_size=128, seed=7)
        self.assertEqual([row["id"] for row in probe_a], [row["id"] for row in probe_b])
        self.assertEqual(len(probe_a), 128)
        self.assertEqual(len(train_a), 12)
        self.assertTrue({row["id"] for row in train_a}.isdisjoint({row["id"] for row in probe_a}))

    def test_training_split_skips_rows_without_labels(self):
        rows = [
            {"problem": f"Problem {index}", "solution": f"Solution \\boxed{{{index}}}"}
            for index in range(140)
        ]
        rows.append({"problem": "bad", "solution": "no final answer here"})
        train, probe = split_math_rows(rows, probe_size=128, seed=7)
        ids = {row["id"] for row in train + probe}
        self.assertFalse(any("bad" in row["problem"] for row in train + probe))
        self.assertEqual(len(ids), 140)

    def test_math500_rows_are_final_only(self):
        rows = normalize_math500_rows([{"problem": "p", "answer": "42"}])
        self.assertEqual(rows[0]["split"], "math500")
        self.assertEqual(rows[0]["source"], "math500")


class ScheduleTests(unittest.TestCase):
    def test_candidate_schedule_shape(self):
        prompt_ids = [f"id-{index}" for index in range(20)]
        schedule = build_candidate_schedule(
            prompt_ids,
            states=48,
            candidates_per_state=16,
            batch_prompts=4,
            seed=3,
        )
        self.assertEqual(len(schedule), 48 * 16)
        self.assertTrue(all(len(candidate.prompt_ids) == 4 for candidate in schedule))
        self.assertTrue(all(len(set(candidate.prompt_ids)) == 4 for candidate in schedule))


class ControllerTests(unittest.TestCase):
    def test_checkpoint_steps_reads_run_default_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "states"
            progress = source / "run_default" / "checkpoints" / "step_1" / "orchestrator" / "progress.pt"
            progress.parent.mkdir(parents=True)
            progress.write_text("ok", encoding="utf-8")

            self.assertEqual(checkpoint_steps(source), [1])

    def test_branch_rl_command_does_not_resume(self):
        args = SimpleNamespace(rl_command="uv run rl")
        command = rl_command(args, Path("branch.toml"))
        self.assertNotIn("--ckpt.resume-step", command)


class PrimeRLConfigTests(unittest.TestCase):
    def test_checkpoint_model_can_use_default_renderer(self):
        config = render_prime_rl_config(
            PrimeRLConfigSpec(
                output_dir=Path("outputs/branch"),
                split_path=Path("outputs/branch/candidate_prompts.jsonl"),
                max_steps=1,
                model_name="/workspace/math_loop_runs/outputs/states/weights/step_1",
                renderer_name="default",
            )
        )
        self.assertIn('name = "/workspace/math_loop_runs/outputs/states/weights/step_1"', config)
        self.assertIn('[orchestrator.renderer]\nname = "default"', config)


if __name__ == "__main__":
    unittest.main()
