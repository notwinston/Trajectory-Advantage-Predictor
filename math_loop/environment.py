"""Local verifiers environment loaded by prime-rl as ``math-loop``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from math_loop.answers import (
    NON_THINKING_SYSTEM_PROMPT,
    exact_match,
    extract_boxed_answer,
)
from math_loop.data import read_jsonl


def load_environment(
    split_path: str,
    *,
    system_prompt: str = NON_THINKING_SYSTEM_PROMPT,
    max_examples: int | None = None,
    **_: Any,
):
    """Build a single-turn boxed-answer reward environment from a JSONL split."""

    import verifiers as vf
    from datasets import Dataset

    rows = read_jsonl(Path(split_path))
    if max_examples is not None:
        rows = rows[:max_examples]
    if not rows:
        raise ValueError(f"no examples found in {split_path}")

    dataset = Dataset.from_list(
        [
            {
                "id": row["id"],
                "question": row.get("question") or row["problem"],
                "answer": row["answer"],
            }
            for row in rows
        ]
    )
    parser = vf.Parser(extract_fn=lambda text: extract_boxed_answer(text, strict=True))

    def boxed_answer_reward(completion, answer, **kwargs):
        parsed = parser.parse_answer(completion) or ""
        return 1.0 if exact_match(parsed, answer) else 0.0

    rubric = vf.Rubric(
        parser=parser,
        funcs=[boxed_answer_reward, parser.get_format_reward_func()],
        weights=[1.0, 0.0],
    )
    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
