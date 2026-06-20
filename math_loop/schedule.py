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


@dataclass(frozen=True)
class TapCandidate:
    """One candidate GRPO batch in the TAP v1 (chain, state, candidate) grid."""

    chain_index: int
    state_index: int
    candidate_index: int
    prompt_ids: tuple[str, ...]

    @property
    def state_id(self) -> str:
        # Matches tap/schema.py: state_id = "{chain}-{state}".
        return f"{self.chain_index}-{self.state_index}"

    @property
    def candidate_id(self) -> str:
        # Matches tap/schema.py: candidate_id = "{state_id}-{k}".
        return f"{self.chain_index}-{self.state_index}-{self.candidate_index}"


def build_tap_schedule(
    prompt_ids: Sequence[str],
    *,
    chains: int = 3,
    states_per_chain: int = 8,
    candidates_per_state: int = 8,
    prompts_per_candidate: int = 2,
    seed: int = 1729,
) -> list[TapCandidate]:
    """Deterministic 2-chain x 6-state x 6-candidate TAP collection schedule.

    Each candidate draws ``prompts_per_candidate`` distinct prompt ids (2 per the
    spec). The schedule is reproducible from ``seed`` and the per-(chain, state)
    RNG stream, so a resumed controller reconstructs the identical grid.
    """
    if chains <= 0:
        raise ValueError("chains must be positive")
    if states_per_chain <= 0:
        raise ValueError("states_per_chain must be positive")
    if candidates_per_state <= 0:
        raise ValueError("candidates_per_state must be positive")
    if prompts_per_candidate <= 0:
        raise ValueError("prompts_per_candidate must be positive")
    unique_ids = list(dict.fromkeys(prompt_ids))
    if len(unique_ids) < prompts_per_candidate:
        raise ValueError(f"need at least {prompts_per_candidate} prompt ids")

    schedule: list[TapCandidate] = []
    for chain_index in range(chains):
        for state_index in range(states_per_chain):
            rng = random.Random(seed + chain_index * 100_000 + state_index)
            for candidate_index in range(candidates_per_state):
                selected = tuple(rng.sample(unique_ids, prompts_per_candidate))
                schedule.append(
                    TapCandidate(
                        chain_index=chain_index,
                        state_index=state_index,
                        candidate_index=candidate_index,
                        prompt_ids=selected,
                    )
                )
    return schedule


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
