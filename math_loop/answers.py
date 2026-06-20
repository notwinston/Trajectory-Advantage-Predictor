"""Answer parsing and prompt helpers for MATH-style boxed answers."""

from __future__ import annotations

import re
from typing import Any


NON_THINKING_SYSTEM_PROMPT = (
    "/no_think\n"
    "Solve the problem briefly. Do not write headings. End with exactly one "
    "final answer in the form \\boxed{...}."
)


def extract_boxed_answer(text: str, *, strict: bool = False) -> str:
    """Return the content of the last ``\\boxed{...}`` expression."""

    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return "" if strict else text

    index = start + len(marker)
    depth = 1
    while index < len(text) and depth:
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        index += 1

    if depth:
        return "" if strict else text
    return text[start + len(marker) : index - 1]


def normalize_answer(value: Any) -> str:
    """Normalize enough for exact-match scoring without symbolic math."""

    text = str(value)
    text = extract_boxed_answer(text, strict=False)
    text = text.strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("$", "")
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".")
    return text


def exact_match(prediction: Any, answer: Any) -> bool:
    return normalize_answer(prediction) == normalize_answer(answer)


def chat_messages(problem: str, system_prompt: str = NON_THINKING_SYSTEM_PROMPT) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]


def render_prompt(tokenizer: Any, problem: str, system_prompt: str = NON_THINKING_SYSTEM_PROMPT) -> str:
    """Render a Qwen-style chat prompt, disabling thinking when supported."""

    messages = chat_messages(problem, system_prompt=system_prompt)
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
