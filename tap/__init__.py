"""TAP v3 — critique-corrected causal-update-utility prediction for GRPO.

v3 synthesizes:
  * v1 (separate, label-efficient GBDT predictor on behavioural GRPO features),
  * v2 (reset-per-cohort battery; redundancy/diversity features),
  * the TAP spec (multi-anchor chains, policy fingerprint, KL-drift penalty,
    within-state selection metrics), and
  * the TAP critiques (GBDT not a 300k-param net on ~100 labels; a *common*
    probe; accuracy as the real target with NLL as a dense proxy; a signal-vs-
    noise gate; selection-lift metrics; gradient-alignment as a *baseline* only).

Design philosophy: predict the **utility of one GRPO update** for held-out MATH,
with a small, monotone-constrained tree/linear model, evaluated by *selection*
quality (ranking among same-anchor candidates) under leave-one-chain-out.

Heavy deps (torch/transformers/peft) live only in ``tap.battery`` and are lazily
imported, so every other module imports and unit-tests on a laptop.
"""

__all__ = [
    "cohorts",
    "features",
    "labels",
    "metrics",
    "predictor",
    "gate",
]
