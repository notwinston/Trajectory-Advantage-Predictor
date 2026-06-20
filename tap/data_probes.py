"""Probe + data bundle for the TAP battery.

Reuses ``math_loop.data`` (MATH split prep) and carves three fixed, disjoint sets:

* ``global``      -- the COMMON probe (same for every candidate at an anchor) for
                     held-out accuracy + teacher-forced NLL.
* ``fingerprint`` -- a small fixed set for the policy-competence fingerprint.
* ``generic``     -- short non-math prompts for the generic-KL drift penalty.

MATH-500 is never touched here (final demonstration only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from math_loop.data import prepare_training_splits, read_jsonl


# Fixed, tiny non-mathematical set for measuring generic drift (KL vs frozen base).
GENERIC_PROMPTS: list[dict[str, Any]] = [
    {"id": "gen-0", "text": "The capital of France is Paris, a city on the river Seine."},
    {"id": "gen-1", "text": "Water boils at one hundred degrees Celsius at sea level."},
    {"id": "gen-2", "text": "Photosynthesis lets plants convert sunlight into chemical energy."},
    {"id": "gen-3", "text": "The novel begins with a long description of the foggy London streets."},
    {"id": "gen-4", "text": "To make tea, boil water and steep the leaves for a few minutes."},
    {"id": "gen-5", "text": "The committee agreed to postpone the meeting until next Thursday."},
    {"id": "gen-6", "text": "A gentle breeze moved through the tall grass on the quiet hillside."},
    {"id": "gen-7", "text": "The software update improved battery life and fixed several bugs."},
]


def _difficulty_filter(rows: list[dict[str, Any]], levels=("Level 3", "Level 4", "Level 5")) -> list[dict[str, Any]]:
    """Keep MATH levels 3-5 when the label is present; otherwise keep all."""

    kept = [r for r in rows if str(r.get("level")) in levels]
    return kept if len(kept) >= max(64, len(rows) // 5) else rows


def prepare_probes(
    data_dir: Path,
    *,
    probe_size: int = 64,
    fingerprint_size: int = 16,
    seed: int = 1729,
    difficulty_filter: bool = True,
) -> dict[str, Any]:
    paths = prepare_training_splits(data_dir, probe_size=probe_size + fingerprint_size, seed=seed)
    train_rows = read_jsonl(paths.train_pool)
    probe_rows = read_jsonl(paths.probe)
    if difficulty_filter:
        train_rows = _difficulty_filter(train_rows)
    global_probe = probe_rows[:probe_size]
    fp_probe = probe_rows[probe_size : probe_size + fingerprint_size]
    return {
        "train_rows": train_rows,
        "probes": {"global": global_probe, "fingerprint": fp_probe, "generic": GENERIC_PROMPTS},
    }
