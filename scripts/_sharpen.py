#!/usr/bin/env python3
"""Try to sharpen the NLL-lift predictor: sweep feature configs, compare
within-domain (fine ranking) vs leave-one-domain-out transfer. Seeds averaged."""
import glob, json, random
from collections import defaultdict
import numpy as np
from tap.predictor import build_xy, LiftPredictor, _oof, MECHANISM_KEYS
from tap.features import FEATURE_KEYS
from tap.metrics import spearman

DIRS = {"science": "outputs/q_science_8x", "codemmlu": "outputs/q_codemmlu",
        "compmath": "outputs/q_compmath", "mmlu": "outputs/ckpt/31645410"}

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
cids = [r.get("candidate_id") for r in rows]

def denoise(d):
    X, y, doms = d["X"], d["y"], d["domains"]
    g = defaultdict(list)
    for i, (dm, c) in enumerate(zip(doms, cids)):
        g[(dm, c)].append(i)
    Xc = np.array([np.nanmean(X[ix], axis=0) for ix in g.values()])
    yc = np.array([float(np.mean(y[ix])) for ix in g.values()])
    dc = np.array([k[0] for k in g.keys()])
    return Xc, yc, dc

def evalcfg(**kw):
    d = build_xy(rows, label="nll", **kw); names = d["names"]
    Xc, yc, dc = denoise(d)
    fp = lambda a, b, c: LiftPredictor(backend="auto", monotone=True).fit(a, b, names=names).predict(c)
    wsp = []
    for dom in DIRS:
        m = dc == dom
        if m.sum() < 10:
            continue
        Xi, yi = Xc[m], yc[m]
        folds = [random.Random(0).randrange(5) for _ in range(len(yi))]
        wsp.append(spearman(_oof(Xi, yi, folds, fp), yi))
    tsp = []
    for held in DIRS:
        te = dc == held
        if te.sum() < 10:
            continue
        mdl = LiftPredictor(backend="auto", monotone=True).fit(Xc[~te], yc[~te], names=names)
        tsp.append(spearman(mdl.predict(Xc[te]), yc[te]))
    return float(np.mean(wsp)), float(np.mean(tsp))

print("%-26s %9s %9s" % ("config", "within", "transfer"))
configs = [
    ("full", dict(with_context=True)),
    ("standardized(per-domain)", dict(with_context=True, standardize_domains=True)),
    ("drop frac_nondegenerate", dict(with_context=True, feature_keys=[k for k in FEATURE_KEYS if k != "frac_nondegenerate"])),
    ("drop frac_nondeg + std", dict(with_context=True, standardize_domains=True, feature_keys=[k for k in FEATURE_KEYS if k != "frac_nondegenerate"])),
    ("mechanism-only", dict(with_context=False, feature_keys=list(MECHANISM_KEYS))),
    ("mechanism-only + std", dict(with_context=False, standardize_domains=True, feature_keys=list(MECHANISM_KEYS))),
]
for name, kw in configs:
    try:
        w, t = evalcfg(**kw)
        print("%-26s %+9.3f %+9.3f" % (name, w, t))
    except Exception as e:
        print("%-26s  ERR %s" % (name, str(e)[:40]))
print("\nMECHANISM_KEYS =", list(MECHANISM_KEYS))
