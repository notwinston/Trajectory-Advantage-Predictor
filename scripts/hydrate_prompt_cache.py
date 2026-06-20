#!/usr/bin/env python3
"""Hydrate dashboard prompt text from source datasets.

The TAP label files intentionally store prompt ids, not full prompt text. This script
reconstructs the domain rows used by the current archived runs and writes a small
local cache that ``build_dashboard_data.py`` can join against.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_loop.data import normalize_math_row
from tap.predictor import load_labels


LABEL_FILES = [
    ROOT / "results" / "doseresp_run1" / "labels.jsonl",
    ROOT / "results" / "labels_archive" / "math_doseresp_n8.jsonl",
    ROOT / "results" / "labels_archive" / "math_qwen25_n115.jsonl",
    ROOT / "results" / "labels_archive" / "science_qwen3_n56.jsonl",
]
OUT = ROOT / "results" / "prompt_cache.jsonl"


def needed_ids() -> set[str]:
    ids: set[str] = set()
    for path in LABEL_FILES:
        if not path.exists():
            continue
        for row in load_labels(path):
            cohort = row.get("cohort") or {}
            ids.update(str(pid) for pid in cohort.get("prompt_ids") or [])
    return ids


def write_cache(rows: list[dict[str, Any]]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with OUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            rid = str(row["id"])
            if rid in seen:
                continue
            seen.add(rid)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def math_rows(target_ids: set[str]) -> list[dict[str, Any]]:
    from datasets import load_dataset

    wanted = {pid for pid in target_ids if pid.startswith("math-train-")}
    if not wanted:
        return []
    out: list[dict[str, Any]] = []
    ds = load_dataset("chiayewken/competition_math", split="train")
    for index, ex in enumerate(ds):
        try:
            row = normalize_math_row(dict(ex), index)
        except ValueError:
            continue
        if row["id"] in wanted:
            out.append(row)
            if len(out) == len(wanted):
                break
    return out


def science_rows(target_ids: set[str], *, seed: int = 0) -> list[dict[str, Any]]:
    from datasets import load_dataset

    wanted = {pid for pid in target_ids if pid.startswith("sciq-")}
    if not wanted:
        return []
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    ds = load_dataset("allenai/sciq", split="train+validation+test")
    for index, ex in enumerate(ds):
        correct = (ex.get("correct_answer") or "").strip()
        distractors = [ex.get("distractor1"), ex.get("distractor2"), ex.get("distractor3")]
        distractors = [d.strip() for d in distractors if d]
        if not correct or len(distractors) < 3:
            continue
        choices = distractors[:3] + [correct]
        rng.shuffle(choices)
        gold = "ABCD"[choices.index(correct)]
        body = ex["question"].strip() + "\n" + "\n".join(f"{letter}) {choice}" for letter, choice in zip("ABCD", choices))
        rid = f"sciq-{index:05d}"
        if rid in wanted:
            out.append(
                {
                    "id": rid,
                    "question": body,
                    "answer": gold,
                    "solution": f"Answer: {gold}",
                    "source": "sciq",
                }
            )
            if len(out) == len(wanted):
                break
    return out


def main() -> None:
    ids = needed_ids()
    rows = math_rows(ids) + science_rows(ids)
    write_cache(rows)
    print(f"needed={len(ids)} hydrated={len(rows)} cache={OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        if exc.name == "datasets":
            raise SystemExit("Missing dependency: install `datasets` to hydrate prompt text.") from None
        raise
