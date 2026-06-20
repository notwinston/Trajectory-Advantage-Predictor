#!/usr/bin/env python3
"""Scale-run analysis: model/generalization stress tests on the larger label set.

Runs three things the n=8 set couldn't support:
  1. backend ablation (ridge vs lightgbm) under LOOCV -- does extra capacity earn
     its keep now that we have more labels?
  2. leave-one-cohort-TYPE-out -- train on some construction methods (e.g.
     label-noise + duplication), predict a HELD-OUT type (e.g. random). Tests
     whether `variance -> lift` is a law or a per-type artifact.
  3. feature importance on the full fit.

Usage:
  python scripts/analyze_scale.py --labels 'outputs/scale_*/labels.jsonl' --label nll
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np

from tap.predictor import build_xy, _oof, _impute, LiftPredictor, conformal_summary
from tap.labels import target_from_row
from tap import metrics as M


def load_many(patterns: list[str]) -> list[dict]:
    rows: list[dict] = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, encoding="utf-8") as h:
                for line in h:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def _kind(r: dict) -> str:
    return (r.get("cohort") or {}).get("kind", "?")


def _fit_predict(backend: str, names):
    def fp(Xtr, ytr, Xte):
        return LiftPredictor(backend=backend, monotone=True).fit(Xtr, ytr, names=names).predict(Xte)
    return fp


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", nargs="+", required=True, help="files or globs")
    ap.add_argument("--label", default="nll", choices=("acc", "nll", "utility"))
    args = ap.parse_args(argv)

    rows = load_many(args.labels)
    rows = [r for r in rows if (v := target_from_row(r, mode=args.label)) is not None and np.isfinite(v)]
    if len(rows) < 5:
        raise SystemExit(f"only {len(rows)} usable labels; need more")
    kinds = [_kind(r) for r in rows]
    print(f"loaded {len(rows)} labels with usable {args.label} target")
    print("cohort types:", dict(Counter(kinds)))

    d = build_xy(rows, label=args.label)
    X, y, names = d["X"], d["y"], d["names"]
    folds = list(range(len(y)))

    print("\n=== backend ablation (LOOCV, all cohorts one anchor) ===")
    grp = ["all"] * len(y)
    for backend in ("ridge", "lightgbm"):
        try:
            pred = _oof(X, y, folds, _fit_predict(backend, names))
            ok = np.isfinite(pred)
            cs = conformal_summary(pred[ok], y[ok])
            print(f"[{backend:9s}] spearman={M.spearman(pred[ok], y[ok]):+.3f}  "
                  f"pairwise={M.pairwise_ranking_accuracy(pred[ok], y[ok], [g for g, m in zip(grp, ok) if m]):.3f}  "
                  f"rmse={M.rmse(pred[ok], y[ok]):.5f}  "
                  f"indet_sign={cs.get('indeterminate_sign_rate', float('nan')):.2f}")
        except Exception as e:  # lightgbm may be absent
            print(f"[{backend:9s}] skipped: {e}")

    print("\n=== leave-one-cohort-TYPE-out (the generalization test) ===")
    for backend in ("ridge", "lightgbm"):
        try:
            pred = _oof(X, y, kinds, _fit_predict(backend, names))   # fold by cohort kind
            ok = np.isfinite(pred)
            print(f"[{backend}] overall: spearman={M.spearman(pred[ok], y[ok]):+.3f}  rmse={M.rmse(pred[ok], y[ok]):.5f}")
            for k in sorted(set(kinds)):
                te = [i for i, kk in enumerate(kinds) if kk == k and np.isfinite(pred[i])]
                if len(te) >= 3 and np.nanstd(y[te]) > 0:
                    print(f"    held-out {k:16s} (n={len(te):3d}): spearman={M.spearman(pred[te], y[te]):+.3f}")
        except Exception as e:
            print(f"[{backend}] skipped: {e}")

    print("\n=== feature importance (ridge, full fit) ===")
    Xi, _ = _impute(X)
    ex = LiftPredictor(backend="ridge").fit(Xi, y, names=names).explain(Xi, top_k=10)
    for f in ex["features"]:
        print(f"   {f['feature']:22s} imp={f['importance']:.4f} dir={f['direction']:+d}")


if __name__ == "__main__":
    main()
