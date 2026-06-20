"""MATH level 3-5 split preparation for TAP v1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Sequence

from math_loop.answers import extract_boxed_answer


DEFAULT_MATH_DATASET = "chiayewken/competition_math"
DEFAULT_MATH_SPLIT = "train"
DEFAULT_MATH500_DATASET = "HuggingFaceH4/MATH-500"
DEFAULT_MATH500_SPLIT = "test"
TAP_LEVELS = frozenset({3, 4, 5})


@dataclass(frozen=True)
class TapSplitPaths:
    train_pool: Path
    heldout_pool: Path
    generic_prompts: Path


@dataclass(frozen=True)
class TapAllSplitPaths(TapSplitPaths):
    math500: Path


GENERIC_DRIFT_PROMPTS = (
    "Write one concise sentence about maintaining a garden.",
    "Explain how to make a cup of tea in two short steps.",
    "Summarize why sleep is useful for learning.",
    "Give a neutral greeting for a new teammate.",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_math_level(value: Any) -> int:
    """Parse MATH level values such as ``"Level 4"`` or ``4``."""

    if isinstance(value, bool):
        raise ValueError("level must be numeric or a Level N string")
    if isinstance(value, int):
        level = value
    else:
        match = re.search(r"\d+", str(value or ""))
        if not match:
            raise ValueError(f"could not parse MATH level from {value!r}")
        level = int(match.group())
    if level not in TAP_LEVELS:
        raise ValueError(f"MATH level {level} is outside TAP levels {sorted(TAP_LEVELS)}")
    return level


def stable_problem_id(prefix: str, index: int, problem: str) -> str:
    digest = hashlib.sha1(problem.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{index:05d}-{digest}"


def normalize_tap_math_row(row: dict[str, Any], index: int, *, prefix: str = "tap-math") -> dict[str, Any]:
    """Normalize one MATH row and require subject/level metadata."""

    if "level" not in row or "type" not in row:
        missing = [key for key in ("level", "type") if key not in row]
        raise ValueError(f"MATH row {index} is missing required TAP fields: {missing}")
    level = parse_math_level(row["level"])
    subject = str(row["type"]).strip()
    if not subject:
        raise ValueError(f"MATH row {index} has an empty subject/type")

    problem = str(row.get("problem") or row.get("question") or "").strip()
    solution = str(row.get("solution") or row.get("completion") or row.get("answer") or "").strip()
    answer = str(row.get("answer") or extract_boxed_answer(solution, strict=True)).strip()
    if not problem:
        raise ValueError(f"MATH row {index} has no problem/question field")
    if not answer:
        raise ValueError(f"MATH row {index} has no answer and no boxed solution")
    return {
        "id": stable_problem_id(prefix, index, problem),
        "source_index": index,
        "source": prefix,
        "problem": problem,
        "question": problem,
        "solution": solution,
        "answer": answer,
        "subject": subject,
        "difficulty": level,
        "level": level,
        "split": "unassigned",
    }


def normalize_tap_math_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            normalized.append(normalize_tap_math_row(row, index))
        except ValueError as exc:
            if "outside TAP levels" in str(exc):
                continue
            raise
    if not normalized:
        raise ValueError("no MATH level 3-5 rows were found")
    return normalized


def split_tap_math_rows(
    rows: Sequence[dict[str, Any]], *, heldout_size: int = 256, seed: int = 1729
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized = normalize_tap_math_rows(rows)
    if len(normalized) <= heldout_size:
        raise ValueError(f"need more than {heldout_size} TAP MATH rows, got {len(normalized)}")
    indices = list(range(len(normalized)))
    random.Random(seed).shuffle(indices)
    heldout_indices = set(indices[:heldout_size])
    train_pool: list[dict[str, Any]] = []
    heldout_pool: list[dict[str, Any]] = []
    for index, row in enumerate(normalized):
        output = dict(row)
        if index in heldout_indices:
            output["split"] = "tap_heldout"
            heldout_pool.append(output)
        else:
            output["split"] = "tap_train_pool"
            train_pool.append(output)
    return train_pool, heldout_pool


def normalize_math500_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    problem = str(row.get("problem") or row.get("question") or "").strip()
    solution = str(row.get("solution") or "").strip()
    answer = str(row.get("answer") or extract_boxed_answer(solution, strict=True)).strip()
    if not problem:
        raise ValueError(f"MATH-500 row {index} has no problem/question field")
    if not answer:
        raise ValueError(f"MATH-500 row {index} has no answer")
    return {
        "id": stable_problem_id("math500", index, problem),
        "source_index": index,
        "source": "math500",
        "problem": problem,
        "question": problem,
        "solution": solution,
        "answer": answer,
        "split": "math500_final",
    }


def normalize_math500_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_math500_row(row, index) for index, row in enumerate(rows)]


def assert_no_math500_leakage(paths: Iterable[Path]) -> None:
    for path in paths:
        for row in read_jsonl(path):
            if str(row.get("source", "")).lower() == "math500" or str(row.get("split", "")).startswith("math500"):
                raise ValueError(f"MATH-500 row leaked into TAP training artifact: {path}")


def _load_hf_rows(dataset_name: str, split: str, dataset_config: str | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    return [dict(row) for row in dataset]


def prepare_tap_training_splits(
    data_dir: Path,
    *,
    math_dataset: str = DEFAULT_MATH_DATASET,
    math_split: str = DEFAULT_MATH_SPLIT,
    math_dataset_config: str | None = None,
    heldout_size: int = 256,
    seed: int = 1729,
    force: bool = False,
) -> TapSplitPaths:
    train_path = data_dir / "tap_math_l3_5_train_pool.jsonl"
    heldout_path = data_dir / f"tap_math_l3_5_heldout{heldout_size}.jsonl"
    generic_path = data_dir / "generic_drift_prompts.jsonl"
    if not force and train_path.exists() and heldout_path.exists() and generic_path.exists():
        assert_no_math500_leakage((train_path, heldout_path))
        return TapSplitPaths(train_pool=train_path, heldout_pool=heldout_path, generic_prompts=generic_path)

    rows = _load_hf_rows(math_dataset, math_split, math_dataset_config)
    train_pool, heldout_pool = split_tap_math_rows(rows, heldout_size=heldout_size, seed=seed)
    write_jsonl(train_path, train_pool)
    write_jsonl(heldout_path, heldout_pool)
    write_jsonl(
        generic_path,
        [{"id": f"generic-{index}", "prompt": prompt, "split": "generic_drift"} for index, prompt in enumerate(GENERIC_DRIFT_PROMPTS)],
    )
    return TapSplitPaths(train_pool=train_path, heldout_pool=heldout_path, generic_prompts=generic_path)


def prepare_tap_final_split(
    data_dir: Path,
    *,
    math500_dataset: str = DEFAULT_MATH500_DATASET,
    math500_split: str = DEFAULT_MATH500_SPLIT,
    math500_dataset_config: str | None = None,
    force: bool = False,
) -> Path:
    final_path = data_dir / "math500_final.jsonl"
    if not force and final_path.exists():
        return final_path
    rows = _load_hf_rows(math500_dataset, math500_split, math500_dataset_config)
    write_jsonl(final_path, normalize_math500_rows(rows))
    return final_path

