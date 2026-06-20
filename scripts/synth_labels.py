#!/usr/bin/env python3
"""Generate a synthetic TAP labels.jsonl with a *known* signal + tunable noise.

Lets you exercise + sanity-check the predictor and the gate end-to-end on a laptop
(no GPU): a planted lift function of the features, repeated across seeds so the
gate has within-cohort noise to measure. Use it to verify the pipeline before
spending pod time.

  python scripts/synth_labels.py --out outputs/synth/labels.jsonl --noise 0.01
  python -m tap.gate --labels outputs/synth/labels.jsonl
  python -m tap.predictor --labels outputs/synth/labels.jsonl --scheme logo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/synth/labels.jsonl"))
    p.add_argument("--chains", type=int, default=3)
    p.add_argument("--anchors", type=int, default=4)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--noise", type=float, default=0.01, help="within-cohort lift noise std")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w", encoding="utf-8") as h:
        for chain in range(args.chains):
            for anchor in range(args.anchors):
                fp_nll = 1.5 - 0.1 * anchor + rng.gauss(0, 0.05)  # weaker early -> higher
                for k in range(args.candidates):
                    # planted cohort features
                    frac_nd = rng.random()
                    redundancy = rng.random()
                    tsim = rng.random()
                    reward_std = 0.5 * frac_nd + rng.uniform(0, 0.1)
                    pass_rate = rng.random()
                    noise_frac = rng.choice([0.0, 0.0, 0.0, 0.25, 0.5, 1.0])
                    # TRUE per-cohort lift (the signal the predictor should recover)
                    true_lift = (0.06 * frac_nd - 0.05 * redundancy + 0.03 * tsim
                                 - 0.07 * noise_frac + 0.02 * (fp_nll - 1.3))
                    for s in range(args.seeds):
                        lift_acc = true_lift + rng.gauss(0, args.noise)
                        lift_nll = 4.0 * true_lift + rng.gauss(0, 4 * args.noise)  # correlated proxy
                        kl_drift = max(0.0, 0.001 * redundancy + rng.gauss(0, 0.0005))
                        rec = {
                            "schema_version": 3,
                            "chain_id": chain,
                            "anchor_index": anchor,
                            "candidate_id": f"c{chain}a{anchor}k{k}",
                            "seed": s,
                            "cohort": {"name": f"c{chain}a{anchor}k{k}", "kind": "synthetic",
                                       "meta": {"label_noise_frac": noise_frac}},
                            "reward_summary": {
                                "reward_mean": pass_rate, "reward_std": reward_std, "pass_rate": pass_rate,
                                "group_passrate_mean": pass_rate, "group_passrate_std": reward_std,
                                "frac_nondegenerate": frac_nd, "frac_all_correct": max(0.0, pass_rate - 0.5),
                                "frac_all_wrong": max(0.0, 0.5 - pass_rate),
                                "mean_logprob": -1.0 - redundancy, "surprisal_mean": 1.0 + redundancy,
                                "entropy_mean": 1.0 - 0.5 * redundancy,
                                "logprob_p10": -2.0, "logprob_p50": -1.0, "logprob_p90": -0.2,
                                "confidence_slope": rng.gauss(0, 0.1),
                                "redundancy_mean": redundancy, "len_mean": 200.0, "len_std": 50.0,
                                "n_groups": 8, "n_rollouts": 64,
                            },
                            "target_similarity": tsim,
                            "fingerprint_nll": fp_nll, "fingerprint_entropy": 1.0,
                            "step_frac": anchor / max(args.anchors - 1, 1),
                            "acc_before": 0.2, "acc_after": 0.2 + lift_acc,
                            "nll_before": 1.0, "nll_after": 1.0 - lift_nll,
                            "kl_before": 0.0, "kl_after": kl_drift,
                            "lift_acc": lift_acc, "lift_nll": lift_nll, "kl_drift": kl_drift,
                            "utility": 100.0 * (lift_acc - 0.05 * kl_drift),
                        }
                        h.write(json.dumps(rec, sort_keys=True) + "\n")
                        n += 1
    print(f"wrote {n} synthetic labels to {args.out}")


if __name__ == "__main__":
    main()
