#!/usr/bin/env python3
"""Predicted vs actual NLL-lift: rank (spearman) + magnitude (pearson, RMSE/std)
for within-group (within-domain CV) and cross-group (leave-one-domain-out).
Full 960-label set, seeds averaged per cohort."""
import glob, json, random
from collections import defaultdict
import numpy as np
from tap.predictor import build_xy, LiftPredictor, _oof
from tap.metrics import spearman, pearson, rmse

DIRS = {"science": "outputs/q_science_8x", "codemmlu": "outputs/q_codemmlu",
        "compmath": "outputs/q_compmath", "mmlu": "outputs/q_mmlu"}

rows, seen = [], set()
for dom, d in DIRS.items():
    for f in glob.glob(d + "/labels_shard_*.jsonl"):
        if "rollouts" in f:
            continue
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("lift_nll") is None:
                continue
            k = (dom, r.get("candidate_id"), r.get("seed"))
            if k in seen:
                continue
            seen.add(k); rows.append(r)

d = build_xy(rows, label="nll", with_context=True)
X, y, doms, names = d["X"], d["y"], d["domains"], d["names"]
cids = [r.get("candidate_id") for r in rows]
g = defaultdict(list)
for i, (dm, c) in enumerate(zip(doms, cids)):
    g[(dm, c)].append(i)
Xc = np.array([np.nanmean(X[ix], axis=0) for ix in g.values()])
yc = np.array([float(np.mean(y[ix])) for ix in g.values()])
dc = np.array([k[0] for k in g.keys()])
jfn = names.index("frac_nondegenerate")

def metrics(pred, yt):
    s = np.std(yt)
    return spearman(pred, yt), pearson(pred, yt), rmse(pred, yt), (rmse(pred, yt) / s if s > 0 else float("nan"))

def gbdt(a, b, c):
    return LiftPredictor(backend="auto", monotone=True).fit(a, b, names=names).predict(c)

def single(a, b, c):  # ridge on the one feature -> calibrated value
    return LiftPredictor(backend="ridge", monotone=False).fit(a[:, [jfn]], b).predict(c[:, [jfn]])

print("actual NLL-lift: mean=%.2f std=%.2f (the scale we're trying to hit)\n" % (yc.mean(), yc.std()))
print("%-22s %8s %8s %8s %9s" % ("setting", "spearman", "pearson", "rmse", "rmse/std"))

# WITHIN-GROUP: per domain 5-fold cohort CV, pooled the OOF preds
for label, fn in [("WITHIN gbdt(23feat)", gbdt), ("WITHIN single-feat", single)]:
    ps, ys = [], []
    for dom in DIRS:
        m = dc == dom
        Xi, yi = Xc[m], yc[m]
        folds = [random.Random(0).randrange(5) for _ in range(len(yi))]
        p = _oof(Xi, yi, folds, fn)
        ok = np.isfinite(p); ps += list(p[ok]); ys += list(yi[ok])
    print("%-22s %+8.3f %+8.3f %8.3f %9.2f" % (label, *metrics(np.array(ps), np.array(ys))))

# CROSS-GROUP: leave-one-domain-out
for label, fn in [("CROSS gbdt(23feat)", gbdt), ("CROSS single-feat", single)]:
    ps, ys = [], []
    for held in DIRS:
        te = dc == held
        p = fn(Xc[~te], yc[~te], Xc[te])
        ps += list(p); ys += list(yc[te])
    print("%-22s %+8.3f %+8.3f %8.3f %9.2f" % (label, *metrics(np.array(ps), np.array(ys))))
