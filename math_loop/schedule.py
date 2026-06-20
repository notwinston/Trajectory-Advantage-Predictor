"""Candidate batch scheduling for branch labels."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence


@dataclass(frozen=True)
class CandidateBatch:
    state_index: int
    candidate_index: int
    prompt_ids: tuple[str, ...]


def build_candidate_schedule(
    prompt_ids: Sequence[str],
    *,
    states: int = 48,
    candidates_per_state: int = 16,
    batch_prompts: int = 4,
    seed: int = 1729,
) -> list[CandidateBatch]:
    if states <= 0:
        raise ValueError("states must be positive")
    if candidates_per_state <= 0:
        raise ValueError("candidates_per_state must be positive")
    if batch_prompts <= 0:
        raise ValueError("batch_prompts must be positive")
    unique_ids = list(dict.fromkeys(prompt_ids))
    if len(unique_ids) < batch_prompts:
        raise ValueError(f"need at least {batch_prompts} prompt ids")

    schedule: list[CandidateBatch] = []
    for state_index in range(1, states + 1):
        rng = random.Random(seed + state_index)
        for candidate_index in range(candidates_per_state):
            selected = tuple(rng.sample(unique_ids, batch_prompts))
            schedule.append(
                CandidateBatch(
                    state_index=state_index,
                    candidate_index=candidate_index,
                    prompt_ids=selected,
                )
            )
    return schedule
