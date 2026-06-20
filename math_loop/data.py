"""Deterministic MATH split creation for the prime-rl loop."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

from math_loop.answers import extract_boxed_answer


DEFAULT_MATH_DATASET = "chiayewken/competition_math"
DEFAULT_MATH_SPLIT = "train"
DEFAULT_MATH500_DATASET = "HuggingFaceH4/MATH-500"
DEFAULT_MATH500_SPLIT = "test"


@dataclass(frozen=True)
class TrainingSplitPaths:
    train_pool: Path
    probe: Path


@dataclass(frozen=True)
class SplitPaths(TrainingSplitPaths):
    math500: Path


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


def stable_problem_id(prefix: str, index: int, problem: str) -> str:
    digest = hashlib.sha1(problem.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{index:05d}-{digest}"


def normalize_math_row(row: dict[str, Any], index: int, *, prefix: str = "math-train") -> dict[str, Any]:
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
    }


def normalize_math500_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    problem = str(row.get("problem") or row.get("question") or "").strip()
    answer = str(row.get("answer") or extract_boxed_answer(str(row.get("solution") or ""), strict=True)).strip()
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
        "solution": str(row.get("solution") or "").strip(),
        "answer": answer,
    }


def split_math_rows(
    rows: Sequence[dict[str, Any]], *, probe_size: int = 128, seed: int = 1729
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) <= probe_size:
        raise ValueError(f"need more than {probe_size} MATH rows, got {len(rows)}")

    normalized = []
    for index, row in enumerate(rows):
        try:  # some MATH rows have no parseable boxed answer; skip rather than abort
            normalized.append(normalize_math_row(row, index))
        except ValueError:
            continue
    indices = list(range(len(normalized)))
    random.Random(seed).shuffle(indices)
    probe_indices = set(indices[:probe_size])

    probe: list[dict[str, Any]] = []
    train_pool: list[dict[str, Any]] = []
    for index, row in enumerate(normalized):
        split_row = dict(row)
        if index in probe_indices:
            split_row["split"] = "probe"
            probe.append(split_row)
        else:
            split_row["split"] = "train_pool"
            train_pool.append(split_row)
    return train_pool, probe


def normalize_math500_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    final_rows = []
    for index, row in enumerate(rows):
        final_row = normalize_math500_row(row, index)
        final_row["split"] = "math500"
        final_rows.append(final_row)
    return final_rows


def _load_hf_rows(dataset_name: str, split: str, dataset_config: str | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    return [dict(row) for row in dataset]


def prepare_training_splits(
    data_dir: Path,
    *,
    math_dataset: str = DEFAULT_MATH_DATASET,
    math_split: str = DEFAULT_MATH_SPLIT,
    math_dataset_config: str | None = None,
    probe_size: int = 128,
    seed: int = 1729,
    force: bool = False,
) -> TrainingSplitPaths:
    train_path = data_dir / "train_pool.jsonl"
    probe_path = data_dir / f"probe{probe_size}.jsonl"
    if not force and train_path.exists() and probe_path.exists():
        return TrainingSplitPaths(train_pool=train_path, probe=probe_path)

    rows = _load_hf_rows(math_dataset, math_split, math_dataset_config)
    train_pool, probe = split_math_rows(rows, probe_size=probe_size, seed=seed)
    write_jsonl(train_path, train_pool)
    write_jsonl(probe_path, probe)
    return TrainingSplitPaths(train_pool=train_path, probe=probe_path)


def prepare_final_split(
    data_dir: Path,
    *,
    math500_dataset: str = DEFAULT_MATH500_DATASET,
    math500_split: str = DEFAULT_MATH500_SPLIT,
    math500_dataset_config: str | None = None,
    force: bool = False,
) -> Path:
    final_path = data_dir / "math500.jsonl"
    if not force and final_path.exists():
        return final_path
    rows = _load_hf_rows(math500_dataset, math500_split, math500_dataset_config)
    write_jsonl(final_path, normalize_math500_rows(rows))
    return final_path


def prepare_all_splits(data_dir: Path, *, force: bool = False, seed: int = 1729) -> SplitPaths:
    training = prepare_training_splits(data_dir, force=force, seed=seed)
    final = prepare_final_split(data_dir, force=force)
    return SplitPaths(train_pool=training.train_pool, probe=training.probe, math500=final)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MATH loop JSONL splits.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/math_loop"))
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--probe-size", type=int, default=128)
    parser.add_argument("--include-final", action="store_true", help="also write MATH-500")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    training = prepare_training_splits(
        args.data_dir,
        seed=args.seed,
        probe_size=args.probe_size,
        force=args.force,
    )
    print(f"train_pool={training.train_pool}")
    print(f"probe={training.probe}")
    if args.include_final:
        print(f"math500={prepare_final_split(args.data_dir, force=args.force)}")


if __name__ == "__main__":
    main()
