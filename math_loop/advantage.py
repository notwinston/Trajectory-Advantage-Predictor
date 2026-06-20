"""Custom DeepSeekMath-style GRPO advantage for prime-rl."""

from __future__ import annotations

import statistics
from typing import Any


def _reward_value(rollout: Any) -> float:
    if isinstance(rollout, dict):
        value = rollout.get("reward", rollout.get("score", 0.0))
    else:
        value = getattr(rollout, "reward", getattr(rollout, "score", 0.0))
    if isinstance(value, dict):
        value = value.get("reward", value.get("score", 0.0))
    return float(value)


def normalized_advantage(inputs: Any, eps: float = 1e-8):
    """Return per-group z-scored rewards, or zeros when the group is flat."""

    from prime_rl.orchestrator.advantage import AdvantageOutputs

    rewards = [_reward_value(rollout) for rollout in inputs.rollouts]
    if not rewards:
        return AdvantageOutputs(advantages=[])

    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    if std <= eps:
        return AdvantageOutputs(advantages=[0.0 for _ in rewards])
    return AdvantageOutputs(advantages=[(reward - mean) / (std + eps) for reward in rewards])
