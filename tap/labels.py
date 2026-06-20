"""TAP utility label — accuracy-first, with NLL as a dense proxy and a KL penalty.

Addresses the TAP-spec label critiques:

* **Accuracy is the real target** (we forecast MATH-500 gains), so ``lift_acc`` is
  the default trained label; ``lift_nll`` (a cheap, dense teacher-forced proxy) and
  the composite ``utility`` are alternates you must *validate against* ``lift_acc``,
  not assume.
* **A single COMMON probe** is used for both ``acc`` and ``nll`` so that candidates
  branched from the same anchor are ranked on the same yardstick (the spec's
  per-candidate "matched probe" broke within-anchor comparison).
* **Weights are explicit and all components are stored**, never folded into one
  opaque magic number; the KL-drift penalty is one-sided (only drift *away* from a
  frozen reference is penalized) and only meaningful once you take >1 step.
* **No sign-ambiguous product** (no ``delta_eval * delta_loss * e^-redundancy``).

All NLL / KL values are means in nats per non-padding token.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class UtilityWeights:
    """Explicit, documented weights for the composite utility (in 'points')."""

    acc: float = 1.0       # weight on held-out accuracy gain (the real target)
    nll: float = 0.0       # weight on probe-NLL gain (dense proxy; off by default)
    kl: float = 0.05       # one-sided penalty on generic drift (nats/token)
    scale: float = 100.0   # readability scale (accuracy gain in points)


@dataclass(frozen=True)
class LiftLabel:
    """All measured components for one branched candidate (common probe)."""

    acc_before: float
    acc_after: float
    nll_before: float          # common probe, teacher-forced NLL (nats/token)
    nll_after: float
    kl_before: float = 0.0     # generic-drift probe: KL(current || frozen ref)
    kl_after: float = 0.0

    @property
    def lift_acc(self) -> float:
        return self.acc_after - self.acc_before

    @property
    def lift_nll(self) -> float:
        # positive = held-out NLL went down = good
        return self.nll_before - self.nll_after

    @property
    def kl_drift(self) -> float:
        # one-sided: only *increases* in generic drift are penalized
        return max(self.kl_after - self.kl_before, 0.0)

    def utility(self, w: UtilityWeights = UtilityWeights()) -> float:
        return w.scale * (w.acc * self.lift_acc + w.nll * self.lift_nll - w.kl * self.kl_drift)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(
            lift_acc=self.lift_acc,
            lift_nll=self.lift_nll,
            kl_drift=self.kl_drift,
            utility=self.utility(),
        )
        return d


# Targets a label row can expose; ``acc`` is the default and the ground truth.
TARGET_MODES = ("acc", "nll", "utility")


def target_from_row(row: dict, mode: str = "acc", weights: UtilityWeights | None = None) -> float | None:
    """Pull the chosen training target from a stored label row.

    Falls back across acc -> nll -> legacy ``delta_probe_nll`` so older v1 label
    files still load.
    """

    if mode not in TARGET_MODES:
        raise ValueError(f"unknown target mode: {mode}")
    if mode == "acc" and row.get("lift_acc") is not None:
        return float(row["lift_acc"])
    if mode == "nll" and row.get("lift_nll") is not None:
        return float(row["lift_nll"])
    if mode == "utility" and row.get("utility") is not None:
        return float(row["utility"])
    if mode == "utility" and {"lift_acc", "lift_nll", "kl_drift"} <= row.keys():
        w = weights or UtilityWeights()
        return w.scale * (w.acc * row["lift_acc"] + w.nll * row["lift_nll"] - w.kl * row.get("kl_drift", 0.0))
    # fallbacks
    if row.get("lift_acc") is not None:
        return float(row["lift_acc"])
    if row.get("lift_nll") is not None:
        return float(row["lift_nll"])
    if row.get("delta_probe_nll") is not None:
        return -float(row["delta_probe_nll"])
    return None
