import unittest

from math_loop.answers import (
    NON_THINKING_SYSTEM_PROMPT,
    exact_match,
    extract_boxed_answer,
)
from math_loop.data import normalize_math500_rows, split_math_rows
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


if __name__ == "__main__":
    unittest.main()
