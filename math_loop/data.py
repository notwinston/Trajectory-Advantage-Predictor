"""Deterministic MATH split creation for the prime-rl loop."""

from __future__ import annotations

import argparse
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


def parse_math_level(value: Any) -> int | None:
    """Parse the trailing integer of a MATH ``level`` field.

    ``chiayewken/competition_math`` stores ``level`` as a string like
    ``"Level 3"``. Return the int (3) or ``None`` when it cannot be parsed
    (e.g. ``"Level ?"`` or missing) so callers can drop the row.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    match = re.search(r"(\d+)\s*$", text)
    if match is None:
        return None
    return int(match.group(1))


def normalize_math_row(row: dict[str, Any], index: int, *, prefix: str = "math-train") -> dict[str, Any]:
    problem = str(row.get("problem") or row.get("question") or "").strip()
    solution = str(row.get("solution") or row.get("completion") or row.get("answer") or "").strip()
    answer = str(row.get("answer") or extract_boxed_answer(solution, strict=True)).strip()
    if not problem:
        raise ValueError(f"MATH row {index} has no problem/question field")
    if not answer:
        raise ValueError(f"MATH row {index} has no answer and no boxed solution")
    subject = str(row.get("type") or row.get("subject") or "unknown").strip() or "unknown"
    return {
        "id": stable_problem_id(prefix, index, problem),
        "source_index": index,
        "source": prefix,
        "problem": problem,
        "question": problem,
        "solution": solution,
        "answer": answer,
        # TAP v1: retain parsed difficulty level (int|None) + subject (str).
        "level": parse_math_level(row.get("level")),
        "subject": subject,
    }


def filter_math_levels(
    rows: Sequence[dict[str, Any]], *, keep: tuple[int, ...] = (3, 4, 5), prefix: str = "math-train"
) -> list[dict[str, Any]]:
    """Normalize ``rows`` and keep only those at MATH levels in ``keep``.

    Rows whose ``level`` cannot be parsed are dropped (never silently coerced).
    """
    keep_set = set(keep)
    kept: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        normalized = normalize_math_row(row, index, prefix=prefix)
        if normalized["level"] in keep_set:
            kept.append(normalized)
    return kept


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


def normalize_math_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize MATH rows, SKIPPING any without a usable problem/answer.

    The real ``chiayewken/competition_math`` split contains rows that lack a
    boxed answer; crashing on them would abort data prep on the pod (this was the
    failure Mark's fresh-branch loop hit). We drop them and report the count.
    """
    normalized: list[dict[str, Any]] = []
    skipped = 0
    for index, row in enumerate(rows):
        try:
            normalized.append(normalize_math_row(row, index))
        except ValueError:
            skipped += 1
    if skipped:
        print(f"Skipping {skipped} MATH rows without usable labels", flush=True)
    return normalized


def split_math_rows(
    rows: Sequence[dict[str, Any]], *, probe_size: int = 128, seed: int = 1729
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized = normalize_math_rows(rows)
    leveled = [row for row in normalized if row.get("level") in (3, 4, 5)]
    if any(row.get("level") is not None for row in normalized):
        normalized = leveled
    if len(normalized) <= probe_size:
        raise ValueError(
            f"need more than {probe_size} valid MATH rows after filtering, got {len(normalized)}"
        )
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


# --- TAP v1 probe sets -------------------------------------------------------
#
# Four fixed probe sets are built from the held-out MATH problems (spec section
# "PROBE SETS"). Matched/global/fingerprint draw from real held-out MATH rows;
# the generic drift probe is four fixed short non-math prompts (constants).

MATCHED_PROBE_SIZE = 16
GLOBAL_PROBE_SIZE = 16
GENERIC_DRIFT_PROBE_SIZE = 8
FINGERPRINT_PROBE_SIZE = 16

# Four short, fixed, non-math prompts used only for generic drift (incremental
# KL). Held constant across every state and chain.
GENERIC_DRIFT_PROMPTS: tuple[str, ...] = (
    "Write a one-sentence description of a sunny day at the beach.",
    "List three common fruits.",
    "Explain what a library is in one short sentence.",
    "Name a color and an animal.",
    "Describe how to pack a small bag for a day trip.",
    "Give two polite ways to end a meeting.",
    "State one reason people label containers.",
    "Write a short reminder to drink water.",
)


@dataclass(frozen=True)
class ProbeSets:
    matched: list[dict[str, Any]]
    global_probe: list[dict[str, Any]]
    generic_drift: list[dict[str, Any]]
    fingerprint: list[dict[str, Any]]


def _sorted_by_id(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: row["id"])


def build_matched_probe(
    probe_rows: Sequence[dict[str, Any]],
    *,
    subject: str | None,
    size: int = MATCHED_PROBE_SIZE,
    seed: int = 1729,
) -> list[dict[str, Any]]:
    """8 held-out MATH problems, same ``subject`` as the candidate when possible.

    Prefers rows whose ``subject`` matches; if fewer than ``size`` exist, fills
    deterministically from the remaining rows. Determinism: a fixed shuffle keyed
    on ``seed``.
    """
    pool = list(probe_rows)
    rng = random.Random(seed)
    if subject is not None:
        same = [row for row in pool if row.get("subject") == subject]
        other = [row for row in pool if row.get("subject") != subject]
        rng.shuffle(same)
        rng.shuffle(other)
        ordered = same + other
    else:
        ordered = list(pool)
        rng.shuffle(ordered)
    return ordered[:size]


def build_global_probe(
    probe_rows: Sequence[dict[str, Any]],
    *,
    size: int = GLOBAL_PROBE_SIZE,
    levels: tuple[int, ...] = (3, 4, 5),
    seed: int = 1729,
) -> list[dict[str, Any]]:
    """8 held-out MATH problems stratified across MATH levels 3-5.

    Round-robins across the requested levels (deterministically shuffled within
    each level) so the set is balanced, then backfills if a level is short.
    """
    rng = random.Random(seed + 1)
    by_level: dict[int, list[dict[str, Any]]] = {level: [] for level in levels}
    leftovers: list[dict[str, Any]] = []
    for row in probe_rows:
        level = row.get("level")
        if level in by_level:
            by_level[level].append(row)
        else:
            leftovers.append(row)
    for bucket in by_level.values():
        rng.shuffle(bucket)
    rng.shuffle(leftovers)

    selected: list[dict[str, Any]] = []
    cursors = {level: 0 for level in levels}
    # Round-robin one row per level until we reach ``size`` or exhaust buckets.
    while len(selected) < size:
        progressed = False
        for level in levels:
            bucket = by_level[level]
            cursor = cursors[level]
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                cursors[level] = cursor + 1
                progressed = True
                if len(selected) >= size:
                    break
        if not progressed:
            break
    if len(selected) < size:
        selected.extend(leftovers[: size - len(selected)])
    return selected[:size]


def build_generic_drift_probe(
    size: int = GENERIC_DRIFT_PROBE_SIZE,
) -> list[dict[str, Any]]:
    """4 fixed short non-math prompts (held constant across all states)."""
    return [
        {"id": f"generic-{index:02d}", "prompt": prompt, "source": "generic_drift"}
        for index, prompt in enumerate(GENERIC_DRIFT_PROMPTS[:size])
    ]


def build_fingerprint_probe(
    probe_rows: Sequence[dict[str, Any]],
    *,
    size: int = FINGERPRINT_PROBE_SIZE,
) -> list[dict[str, Any]]:
    """16 FIXED held-out MATH prompts (identical across every state and chain).

    Selected as the first ``size`` rows by stable id sort so the fingerprint
    probe is reproducible and shared by all policy states.
    """
    return _sorted_by_id(probe_rows)[:size]


def build_probe_sets(
    probe_rows: Sequence[dict[str, Any]],
    *,
    subject: str | None = None,
    seed: int = 1729,
) -> ProbeSets:
    """Assemble all four TAP v1 probe sets from the held-out MATH ``probe_rows``."""
    return ProbeSets(
        matched=build_matched_probe(probe_rows, subject=subject, seed=seed),
        global_probe=build_global_probe(probe_rows, seed=seed),
        generic_drift=build_generic_drift_probe(),
        fingerprint=build_fingerprint_probe(probe_rows),
    )


def _load_hf_rows(dataset_name: str, split: str, dataset_config: str | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    return [dict(row) for row in dataset]


def assert_no_math500_leakage(paths: Iterable[Path]) -> None:
    """Fail closed if a TAP training/probe artifact contains MATH-500 rows."""
    for path in paths:
        for row in read_jsonl(path):
            source = str(row.get("source", "")).lower()
            split = str(row.get("split", "")).lower()
            if source == "math500" or split.startswith("math500"):
                raise ValueError(f"MATH-500 row leaked into TAP training artifact: {path}")


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
        assert_no_math500_leakage((train_path, probe_path))
        return TrainingSplitPaths(train_pool=train_path, probe=probe_path)

    rows = _load_hf_rows(math_dataset, math_split, math_dataset_config)
    train_pool, probe = split_math_rows(rows, probe_size=probe_size, seed=seed)
    write_jsonl(train_path, train_pool)
    write_jsonl(probe_path, probe)
    assert_no_math500_leakage((train_path, probe_path))
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
    assert_no_math500_leakage((training.train_pool, training.probe))
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
