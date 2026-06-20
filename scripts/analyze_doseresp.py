#!/usr/bin/env python3
"""Dose-response analysis for the label-noise battery.

The label-noise cohorts are a *built-in ground truth*: as the corrupted fraction
rises, the reward signal points the policy at wrong answers, so held-out lift should
fall (and go negative). This script checks that monotone relationship, which validates
that (a) the battery measures real lift and (b) our cheap features track data quality.

Usage:
  python scripts/analyze_doseresp.py --labels outputs/tap_doseresp/labels.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from tap import metrics as M


def _frac(row: dict) -> float | None:
    meta = (row.get("cohort") or {}).get("meta") or {}
    if "label_noise_frac" in meta:
        return float(meta["label_noise_frac"])
    cid = str(row.get("candidate_id", ""))
    if cid.startswith("labelnoise-"):
        try:
            return int(cid.split("-")[1]) / 100.0
        except (IndexError, ValueError):
            return None
    return None


def load(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    args = p.parse_args(argv)

    rows = load(args.labels)
    dose = [(f, r) for r in rows if (f := _frac(r)) is not None]
    print(f"loaded {len(rows)} labels; {len(dose)} are label-noise cohorts\n")
    if not dose:
        print("no label-noise cohorts found.")
        return

    by_frac: dict[float, list[dict]] = defaultdict(list)
    for f, r in dose:
        by_frac[f].append(r)

    hdr = f"{'noise_frac':>10} {'n':>3} {'lift_nll(mean±sd)':>22} {'mean_reward':>12} {'len_mean':>9} {'reward_std':>10} {'pass_rate':>9}"
    print(hdr)
    print("-" * len(hdr))
    for f in sorted(by_frac):
        rs = by_frac[f]
        ln = [r.get("lift_nll", 0.0) for r in rs]
        mr = [r.get("mean_reward", 0.0) for r in rs]
        summ = [r.get("reward_summary") or {} for r in rs]
        lm = [s.get("len_mean", 0.0) for s in summ]
        rstd = [s.get("reward_std", 0.0) for s in summ]
        pr = [s.get("pass_rate", 0.0) for s in summ]
        sd = pstdev(ln) if len(ln) > 1 else 0.0
        print(f"{f:>10.2f} {len(rs):>3} {mean(ln):>+11.4f}±{sd:<9.4f} "
              f"{mean(mr):>12.3f} {mean(lm):>9.1f} {mean(rstd):>10.3f} {mean(pr):>9.3f}")

    fracs = [f for f, _ in dose]
    lifts = [r.get("lift_nll", 0.0) for _, r in dose]
    rho = M.spearman(fracs, lifts)
    print(f"\nSpearman(noise_frac, lift_nll) = {rho:+.3f}   "
          f"(want NEGATIVE: more corruption -> less lift)")
    # crude monotonicity over the per-frac means
    means = [mean([r.get("lift_nll", 0.0) for r in by_frac[f]]) for f in sorted(by_frac)]
    mono = all(a >= b - 1e-9 for a, b in zip(means, means[1:]))
    print(f"per-frac mean lift_nll monotone non-increasing: {mono}")
    if rho < -0.3:
        print("=> dose-response CONFIRMED: cheap reward signal tracks data quality.")
    else:
        print("=> weak/absent dose-response: raise LR/steps/cohort-size or check reward.")


if __name__ == "__main__":
    main()
