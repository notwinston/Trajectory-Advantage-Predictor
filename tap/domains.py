"""Pluggable RLVR domains for the TAP battery (math / code / science).

The battery is domain-agnostic: cohorts, features, GRPO, and the predictor only
ever see rewards + log-probs. A ``Domain`` supplies the four things that *are*
domain-specific:

  * ``load_splits``  -- train pool + disjoint probe sets (global/fingerprint/generic).
  * ``reward``       -- verifier: completion text + item -> {0,1} (or fraction).
  * ``render``       -- chat prompt (generic; Qwen3 non-thinking by default).
  * ``target_text``  -- per-domain corpus for ``target_similarity`` (NOT math-anchored).

This is the multi-domain generalization: one generalist policy (e.g. Qwen3-1.7B)
labelled across math+code+science so the predictor learns the domain-invariant
mechanism (reward variance -> lift), per the cross-domain transfer analysis.

Heavy deps (datasets) are imported lazily so this module is importable on a laptop.
"""

from __future__ import annotations

import hashlib
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Protocol


# ---- generic chat render (Qwen3 non-thinking; falls back for other templates) --

def chat_render(tok, question: str, system: str) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": question})
    try:  # Qwen3 supports enable_thinking=False to skip the <think> block
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _unigrams(texts: list[str]) -> dict[str, float]:
    from collections import Counter
    c: Counter = Counter()
    for t in texts:
        c.update(re.findall(r"[a-z0-9]+", (t or "").lower()))
    n = sum(c.values()) or 1
    return {w: v / n for w, v in c.items()}


def _split_ids(n: int, seed: int) -> list[int]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return idx


class Domain(Protocol):
    name: str
    system: str
    def load_splits(self, data_dir: Path, *, probe_size: int, fingerprint_size: int,
                    seed: int) -> dict[str, Any]: ...
    def reward(self, text: str, item: dict) -> float: ...


# ---- MATH (reuses math_loop) --------------------------------------------------

class MathDomain:
    name = "math"
    system = "Solve the problem. Put the final answer in \\boxed{}."

    def load_splits(self, data_dir, *, probe_size, fingerprint_size, seed):
        from math_loop.data import prepare_training_splits, read_jsonl
        paths = prepare_training_splits(data_dir, probe_size=probe_size + fingerprint_size, seed=seed)
        train = read_jsonl(paths.train_pool)
        probe = read_jsonl(paths.probe)
        for r in train + probe:
            r.setdefault("question", r.get("problem", ""))
        return {"train_rows": train,
                "probes": {"global": probe[:probe_size],
                           "fingerprint": probe[probe_size:probe_size + fingerprint_size],
                           "generic": GENERIC_PROMPTS}}

    def reward(self, text, item):
        from math_loop.answers import exact_match, extract_boxed_answer
        return 1.0 if exact_match(extract_boxed_answer(text, strict=True), item.get("answer", "")) else 0.0


# ---- CODE (MBPP; execution-based reward) --------------------------------------

class CodeDomain:
    name = "code"
    system = "You are an expert Python programmer. Write a correct solution inside a single ```python code block."

    def load_splits(self, data_dir, *, probe_size, fingerprint_size, seed):
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/mbpp", "full", split="train+test+validation+prompt")
        rows = []
        for ex in ds:
            tests = list(ex.get("test_list") or [])
            if not tests:
                continue
            rid = "mbpp-%05d" % int(ex.get("task_id", len(rows)))
            q = (ex["text"].strip() + "\nYour function must pass these tests:\n" + "\n".join(tests))
            ref = ex.get("code", "") or ""
            rows.append({"id": rid, "question": q, "answer": "",
                         "solution": f"```python\n{ref}\n```",  # gold completion for NLL-lift
                         "tests": tests, "setup": ex.get("test_setup_code", "") or ""})
        order = _split_ids(len(rows), seed)
        rows = [rows[i] for i in order]
        k = probe_size + fingerprint_size
        probe, train = rows[:k], rows[k:]
        return {"train_rows": train,
                "probes": {"global": probe[:probe_size],
                           "fingerprint": probe[probe_size:k],
                           "generic": GENERIC_PROMPTS}}

    def reward(self, text, item):
        code = _extract_code(text)
        if not code:
            return 0.0
        return _run_pytests(code, item.get("tests", []), item.get("setup", ""))


def _extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len)
    # fall back: take from first def/import to end
    m = re.search(r"(?:^|\n)(import |from |def |class )", text)
    return text[m.start():] if m else text


def _run_pytests(code: str, tests: list[str], setup: str, timeout: int = 6) -> float:
    """All-or-nothing: 1.0 if the code + every assert runs cleanly, else 0.0."""
    if not tests:
        return 0.0
    program = "\n".join([setup, code, *tests])
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(program)
            path = f.name
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        ok = r.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        ok = False
    finally:
        try:
            Path(path).unlink()
        except Exception:
            pass
    return 1.0 if ok else 0.0


# ---- SCIENCE (SciQ; MCQ, exact-letter reward) ---------------------------------

class ScienceDomain:
    name = "science"
    system = "Answer the multiple-choice science question. End with 'Answer: <letter>'."

    def load_splits(self, data_dir, *, probe_size, fingerprint_size, seed):
        from datasets import load_dataset
        ds = load_dataset("allenai/sciq", split="train+validation+test")
        rows = []
        rng = random.Random(seed)
        for i, ex in enumerate(ds):
            correct = (ex.get("correct_answer") or "").strip()
            distractors = [ex.get("distractor1"), ex.get("distractor2"), ex.get("distractor3")]
            distractors = [d.strip() for d in distractors if d]
            if not correct or len(distractors) < 3:
                continue
            choices = distractors[:3] + [correct]
            rng.shuffle(choices)
            gold = "ABCD"[choices.index(correct)]
            body = ex["question"].strip() + "\n" + "\n".join(
                f"{l}) {c}" for l, c in zip("ABCD", choices))
            rows.append({"id": "sciq-%05d" % i, "question": body, "answer": gold,
                         "solution": f"Answer: {gold}"})  # gold completion for NLL-lift
        order = _split_ids(len(rows), seed)
        rows = [rows[i] for i in order]
        k = probe_size + fingerprint_size
        probe, train = rows[:k], rows[k:]
        return {"train_rows": train,
                "probes": {"global": probe[:probe_size],
                           "fingerprint": probe[probe_size:k],
                           "generic": GENERIC_PROMPTS}}

    def reward(self, text, item):
        return 1.0 if _extract_letter(text) == item.get("answer", "") else 0.0


def _extract_letter(text: str) -> str | None:
    m = re.findall(r"[Aa]nswer\b\W*([A-D])", text)
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([A-D])\b", text)
    return m[-1].upper() if m else None


class MMLUDomain:
    name = "mmlu"
    system = "Answer the multiple-choice question. End with 'Answer: <letter>'."

    def load_splits(self, data_dir, *, probe_size, fingerprint_size, seed):
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")
        rows = []
        for i, ex in enumerate(ds):
            ch = ex.get("choices") or []
            ans = ex.get("answer")
            if len(ch) != 4 or ans is None or not (0 <= int(ans) < 4):
                continue
            gold = "ABCD"[int(ans)]
            body = ex["question"].strip() + "\n" + "\n".join(
                f"{l}) {c}" for l, c in zip("ABCD", ch))
            rows.append({"id": "mmlu-%05d" % i, "question": body, "answer": gold,
                         "solution": f"Answer: {gold}"})
        order = _split_ids(len(rows), seed)
        rows = [rows[i] for i in order]
        k = probe_size + fingerprint_size
        probe, train = rows[:k], rows[k:]
        return {"train_rows": train,
                "probes": {"global": probe[:probe_size],
                           "fingerprint": probe[probe_size:k],
                           "generic": GENERIC_PROMPTS}}

    def reward(self, text, item):
        return 1.0 if _extract_letter(text) == item.get("answer", "") else 0.0


# Shared generic (non-target) prompts for the KL-drift probe -- domain-neutral text.
GENERIC_PROMPTS: list[dict[str, Any]] = [
    {"id": "gen-0", "text": "The capital of France is Paris, a city on the river Seine."},
    {"id": "gen-1", "text": "Water boils at one hundred degrees Celsius at sea level."},
    {"id": "gen-2", "text": "A gentle breeze moved through the tall grass on the quiet hillside."},
    {"id": "gen-3", "text": "The committee agreed to postpone the meeting until next Thursday."},
    {"id": "gen-4", "text": "To make tea, boil water and steep the leaves for a few minutes."},
    {"id": "gen-5", "text": "The software update improved battery life and fixed several bugs."},
]


DOMAINS: dict[str, Domain] = {"math": MathDomain(), "code": CodeDomain(),
                              "science": ScienceDomain(), "mmlu": MMLUDomain()}


def get_domain(name: str) -> Domain:
    if name not in DOMAINS:
        raise ValueError(f"unknown domain {name!r}; choose from {sorted(DOMAINS)}")
    return DOMAINS[name]
