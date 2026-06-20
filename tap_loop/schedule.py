"""TAP v1 candidate scheduling and history helpers."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence


@dataclass(frozen=True)
class TapCandidateBatch:
    chain_id: int
    state_index: int
    candidate_index: int
    prompt_ids: tuple[str, ...]

    @property
    def state_id(self) -> str:
        return f"chain_{self.chain_id:02d}_state_{self.state_index:03d}"

    @property
    def candidate_id(self) -> str:
        return f"{self.state_id}_candidate_{self.candidate_index:02d}"


def build_tap_candidate_schedule(
    prompt_ids: Sequence[str],
    *,
    chains: int = 2,
    states_per_chain: int = 6,
    candidates_per_state: int = 6,
    batch_prompts: int = 2,
    seed: int = 1729,
) -> list[TapCandidateBatch]:
    if chains <= 0:
        raise ValueError("chains must be positive")
    if states_per_chain <= 0:
        raise ValueError("states_per_chain must be positive")
    if candidates_per_state <= 0:
        raise ValueError("candidates_per_state must be positive")
    if batch_prompts <= 0:
        raise ValueError("batch_prompts must be positive")
    unique_ids = list(dict.fromkeys(prompt_ids))
    if len(unique_ids) < batch_prompts:
        raise ValueError(f"need at least {batch_prompts} prompt ids")

    schedule: list[TapCandidateBatch] = []
    for chain_id in range(chains):
        for state_index in range(states_per_chain):
            rng = random.Random(seed + 10_000 * chain_id + state_index)
            for candidate_index in range(candidates_per_state):
                schedule.append(
                    TapCandidateBatch(
                        chain_id=chain_id,
                        state_index=state_index,
                        candidate_index=candidate_index,
                        prompt_ids=tuple(rng.sample(unique_ids, batch_prompts)),
                    )
                )
    return schedule


def select_main_candidate(candidates_per_state: int, *, chain_id: int, state_index: int, seed: int = 1729) -> int:
    if candidates_per_state <= 0:
        raise ValueError("candidates_per_state must be positive")
    rng = random.Random(seed + 1_000_000 + 10_000 * chain_id + state_index)
    return rng.randrange(candidates_per_state)


def latest_history(candidate_ids: Sequence[str], *, window: int = 4) -> list[str]:
    if window <= 0:
        raise ValueError("history window must be positive")
    return list(candidate_ids[-window:])

