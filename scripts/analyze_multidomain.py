#!/usr/bin/env python3
"""Multi-domain transfer analysis (the headline cross-domain result).

Pools math+code+science labels (same generalist policy) and runs:
  * per-domain summary (baseline competence, lift distribution, saturation check),
  * leave-one-DOMAIN-out generalization (train on 2 domains, predict the 3rd),
  * the TRANSFER LADDER the analysis prescribed:
      raw-all  ->  mechanism-only  ->  standardized-all  ->  mechanism+standardized
    so we can see whether dropping domain-scaled confounds + per-domain
    standardization recovers cross-domain transfer.

Usage:
  python scripts/analyze_multidomain.py --labels 'outputs/*/labels.jsonl' --label nll
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter

import numpy as np

from tap.predictor import build_xy, _oof, LiftPredictor, MECHANISM_KEYS
from tap.features import FEATURE_KEYS
from tap.labels import target_from_row
from tap import metrics as M


def load_many(patterns: list[str]) -> list[dict]:
    rows: list[dict] = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            with open(p, encoding="utf-8") as h:
                for ln in h:
                    ln = ln.strip()
                    if ln:
                        rows.append(json.loads(ln))
    return rows


def _fp(backend: str, names):
    def f(Xtr, ytr, Xte):
        return LiftPredictor(backend=backend, monotone=True).fit(Xtr, ytr, names=names).predict(Xte)
    return f


def leave_one_domain_out(rows, *, label, feature_keys, standardize, backend):
    d = build_xy(rows, label=label, feature_keys=list(feature_keys), standardize_domains=standardize)
    X, y, names, domains = d["X"], d["y"], d["names"], d["domains"]
    pred = _oof(X, y, domains, _fp(backend, names))     # fold by DOMAIN
    ok = np.isfinite(pred)
    overall = M.spearman(pred[ok], y[ok])
    per = {}
    for g in sorted(set(domains)):
        te = [i for i, dd in enumerate(domains) if dd == g and np.isfinite(pred[i])]
        if len(te) >= 3 and np.nanstd(y[te]) > 0:
            per[g] = round(float(M.spearman(pred[te], y[te])), 3)
    return overall, per


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--label", default="nll", choices=("acc", "nll", "utility"))
    ap.add_argument("--backend", default="ridge")
    args = ap.parse_args(argv)

    rows = load_many(args.labels)
    rows = [r for r in rows if target_from_row(r, mode=args.label) is not None]
    doms = [r.get("domain", "?") for r in rows]
    print(f"{len(rows)} labels; domains: {dict(Counter(doms))}")

    print("\n=== per-domain summary (saturation / signal check) ===")
    for g in sorted(set(doms)):
        sub = [r for r in rows if r.get("domain", "?") == g]
        accb = np.nanmean([r.get("acc_before", np.nan) for r in sub])
        lifts = [target_from_row(r, mode=args.label) for r in sub]
        nz = sum(1 for v in lifts if abs(v) > 1e-6)
        print(f"  {g:9s} n={len(sub):3d}  acc_before~{accb:.2f}  nonzero_lift={nz}/{len(sub)}  "
              f"lift=[{min(lifts):+.4f},{max(lifts):+.4f}]")

    if len(set(doms)) < 2:
        print("\n(only one domain present -> leave-one-domain-out needs >=2 domains)")
        return

    print("\n=== leave-one-DOMAIN-out TRANSFER LADDER (held-out-domain Spearman) ===")
    for nm, keys, std in [("raw-all", FEATURE_KEYS, False),
                          ("mechanism-only", MECHANISM_KEYS, False),
                          ("standardized-all", FEATURE_KEYS, True),
                          ("mechanism+standardized", MECHANISM_KEYS, True)]:
        try:
            ov, per = leave_one_domain_out(rows, label=args.label, feature_keys=keys,
                                           standardize=std, backend=args.backend)
            print(f"[{nm:24s}] overall={ov:+.3f}  per-held-out-domain={per}")
        except Exception as e:
            print(f"[{nm:24s}] failed: {e}")


if __name__ == "__main__":
    main()
