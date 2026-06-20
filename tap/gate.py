"""Signal-vs-noise gate — the experiment that decides whether TAP is even possible.

The TAP critique: "nonzero label variance" is far too weak a bar; a one-step lift
can be all noise. Before building hundreds of labels, run a few cohorts with
multiple seeds and check that **between-cohort signal exceeds within-cohort noise**.

We report, per target (acc / nll / utility):
* within-cohort std  (noise: same cohort, different seeds),
* between-cohort std (signal: spread of cohort means),
* ICC = var_between / (var_between + var_within)  (1 = all signal, 0 = all noise),
* SNR = std_between / std_within,
and a verdict. Needs labels with >1 seed per cohort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tap.labels import target_from_row
from tap.predictor import load_labels


def _cohort_key(row: dict[str, Any]) -> str:
    c = row.get("cohort") or {}
    return str(c.get("name") or row.get("candidate_id") or row.get("cohort_name"))


def analyze(rows: Sequence[dict[str, Any]], *, label: str = "acc", min_seeds: int = 2) -> dict[str, Any]:
    by: dict[str, list[float]] = {}
    for row in rows:
        v = target_from_row(row, mode=label)
        if v is None:
            continue
        by.setdefault(_cohort_key(row), []).append(float(v))

    repeated = {k: v for k, v in by.items() if len(v) >= min_seeds}
    means = np.array([np.mean(v) for v in by.values()]) if by else np.array([])

    within_var = float("nan")
    if repeated:
        # pooled within-cohort variance (averaged over cohorts that have repeats)
        within_var = float(np.mean([np.var(v, ddof=1) for v in repeated.values()]))
    between_var = float(np.var(means, ddof=1)) if len(means) > 1 else float("nan")

    icc = snr = float("nan")
    if np.isfinite(within_var) and np.isfinite(between_var) and (within_var + between_var) > 0:
        icc = between_var / (within_var + between_var)
        snr = (between_var ** 0.5) / ((within_var ** 0.5) + 1e-12)

    verdict = "insufficient_repeats"
    if np.isfinite(icc):
        verdict = "strong_signal" if icc >= 0.5 else ("usable_signal" if icc >= 0.2 else "mostly_noise")

    return {
        "label": label,
        "n_cohorts": len(by),
        "n_cohorts_with_repeats": len(repeated),
        "within_cohort_std": within_var ** 0.5 if np.isfinite(within_var) else float("nan"),
        "between_cohort_std": between_var ** 0.5 if np.isfinite(between_var) else float("nan"),
        "icc": icc,
        "snr": snr,
        "verdict": verdict,
        "advice": {
            "strong_signal": "Proceed to full label collection.",
            "usable_signal": "Proceed, but increase grpo_steps / probe_k / cohort_size to raise SNR.",
            "mostly_noise": "Do NOT scale yet: increase steps/probe/cohort size, or the label carries no signal.",
            "insufficient_repeats": "Re-run the battery with --seeds >= 2 per cohort.",
        }[verdict],
    }


def proxy_correlation(rows: Sequence[dict[str, Any]]) -> float:
    """corr(lift_nll, lift_acc): is the dense NLL proxy aligned with the real target?"""

    pairs = [(r["lift_acc"], r["lift_nll"]) for r in rows
             if r.get("lift_acc") is not None and r.get("lift_nll") is not None]
    if len(pairs) < 2:
        return float("nan")
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--min-seeds", type=int, default=2)
    args = p.parse_args(argv)
    rows = load_labels(args.labels)
    report = {lab: analyze(rows, label=lab, min_seeds=args.min_seeds) for lab in ("acc", "nll", "utility")}
    # Decide on EITHER acc or nll (nll is dense; acc is quantized on small probes).
    proceed = any(report[l]["verdict"] in ("usable_signal", "strong_signal") for l in ("acc", "nll"))
    report["decision"] = {
        "proceed": proceed,
        "rule": "proceed if acc OR nll ICC >= 0.2 (nll is the dense early indicator)",
        "proxy_corr_acc_nll": proxy_correlation(rows),
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
