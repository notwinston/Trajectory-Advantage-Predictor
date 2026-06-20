#!/usr/bin/env python3
"""Leave-one-domain-out transfer test on NLL-lift, with honest stats.

Seeds are averaged per cohort (denoise => independent samples => correct n for
significance). For each held-out domain: train on the other 3, predict the
held-out one. Report transfer spearman + p-value + a usefulness metric
(selection lift: actual lift of the top-25%-predicted cohorts vs the mean).
"""
import glob, json, math, random
from collections import defaultdict

import numpy as np
from tap.predictor import build_xy, LiftPredictor, _oof
import tap.metrics as M

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
            seen.add(k)
            r["_dom"] = dom
            rows.append(r)

d = build_xy(rows, with_context=True, label="nll")
X, y, domains, names = d["X"], d["y"], d["domains"], d["names"]
cids = [r.get("candidate_id") for r in rows]

# average seeds per (domain, cohort) -> denoised, independent samples
groups = defaultdict(list)
for i, (dom, c) in enumerate(zip(domains, cids)):
    groups[(dom, c)].append(i)
Xc, yc, domc = [], [], []
for (dom, c), idx in groups.items():
    Xc.append(np.nanmean(X[idx], axis=0)); yc.append(float(np.mean(y[idx]))); domc.append(dom)
Xc, yc, domc = np.array(Xc), np.array(yc), np.array(domc)
print("denoised cohorts/domain:", {dm: int((domc == dm).sum()) for dm in DIRS})

def spear_p(rho, n):
    if n < 4 or abs(rho) >= 1:
        return float("nan")
    t = rho * math.sqrt((n - 2) / max(1 - rho * rho, 1e-9))
    return math.erfc(abs(t) / math.sqrt(2))   # normal approx, 2-sided

def within_cv(mask):
    Xi, yi = Xc[mask], yc[mask]
    folds = [random.Random(0).randrange(5) for _ in range(len(yi))]
    fp = lambda a, b, c: LiftPredictor(backend="auto", monotone=True).fit(a, b, names=names).predict(c)
    p = _oof(Xi, yi, folds, fp)
    return M.spearman(p, yi)

print("\nleave-one-domain-out TRANSFER (NLL), seeds averaged:")
print(f"  {'held-out':9s} {'n':>4s}  {'within':>7s}  {'transfer':>8s}  {'p(transf)':>9s}  {'sel_lift(top25%)':>16s}")
for held in DIRS:
    te = domc == held; tr = ~te
    n = int(te.sum())
    if n < 5:
        continue
    model = LiftPredictor(backend="auto", monotone=True).fit(Xc[tr], yc[tr], names=names)
    pred = model.predict(Xc[te]); yt = yc[te]
    sp = M.spearman(pred, yt)
    within = within_cv(te)
    k = max(1, int(0.25 * n))
    top = np.argsort(-pred)[:k]
    sel = float(np.mean(yt[top]) - np.mean(yt))
    print(f"  {held:9s} {n:4d}  {within:+7.3f}  {sp:+8.3f}  {spear_p(sp, n):9.3g}  "
          f"{sel:+8.3f} (data std {np.std(yt):.2f})")
