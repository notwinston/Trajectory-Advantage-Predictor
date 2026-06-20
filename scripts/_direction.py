#!/usr/bin/env python3
"""Directional agreement: how often sign(predicted NLL-lift) == sign(actual NLL-lift).
Predictor = leave-one-out linear fit of lift_nll ~ adv_std within domain (needs an
intercept to define the zero-crossing). Reports base rate, accuracy, and
sensitivity/specificity (raw accuracy is inflated by base rate)."""
import json, numpy as np
from collections import defaultdict

rows = [json.loads(l) for l in open("outputs/enriched_labels.jsonl")]
DOMS = ["science", "codemmlu", "compmath", "mmlu"]

def adv(r):
    v = (r.get("features") or {}).get("adv_std")
    return float(v) if v is not None else np.nan

g = defaultdict(list)
for r in rows:
    g[(r["domain"], r["candidate_id"])].append(r)
keys = list(g.keys())
dom = np.array([k[0] for k in keys])
Y = np.array([np.mean([r["lift_nll"] for r in g[k]]) for k in keys])
A = np.array([np.nanmean([adv(r) for r in g[k]]) for k in keys])

def loo_pred(x, y):
    """leave-one-out simple linear regression predictions."""
    n = len(y); pred = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        xs, ys = x[m], y[m]
        b1 = np.cov(xs, ys, bias=True)[0, 1] / (np.var(xs) + 1e-12)
        b0 = ys.mean() - b1 * xs.mean()
        pred[i] = b0 + b1 * x[i]
    return pred

def stats(yp, yt, eps=0.0):
    m = np.abs(yt) > eps
    yp, yt = yp[m], yt[m]
    acc = np.mean(np.sign(yp) == np.sign(yt))
    pos = yt > 0; neg = yt < 0
    sens = np.mean(yp[pos] > 0) if pos.any() else float("nan")
    spec = np.mean(yp[neg] < 0) if neg.any() else float("nan")
    return acc, sens, spec, pos.mean(), m.sum()

print("INTRA-DOMAIN directional accuracy (LOO linear on adv_std):")
print("%-10s %7s %8s %8s %8s %6s" % ("domain", "base+", "dir-acc", "sens(+)", "spec(-)", "n"))
allyp, allyt, alldom = [], [], []
for dmn in DOMS:
    mm = dom == dmn
    x, y = A[mm], Y[mm]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    yp = loo_pred(x, y)
    acc, sens, spec, base, n = stats(yp, y)
    allyp.append(yp); allyt.append(y); alldom += [dmn] * len(y)
    print("%-10s %6.0f%% %7.0f%% %7.0f%% %7.0f%% %6d" % (dmn, base * 100, acc * 100, sens * 100, spec * 100, n))
yp = np.concatenate(allyp); yt = np.concatenate(allyt)
acc, sens, spec, base, n = stats(yp, yt)
print("%-10s %6.0f%% %7.0f%% %7.0f%% %7.0f%% %6d" % ("POOLED", base * 100, acc * 100, sens * 100, spec * 100, n))

print("\nExcluding near-zero actual lifts (|actual| > 0.1, where direction is meaningful):")
acc, sens, spec, base, n = stats(yp, yt, eps=0.1)
print("  dir-acc %.0f%%  sens(+) %.0f%%  spec(-) %.0f%%  (n=%d of %d)" % (acc * 100, sens * 100, spec * 100, n, len(yt)))

print("\nbaseline (always predict majority sign):")
print("  pooled: %.0f%%" % (100 * max(np.mean(yt > 0), np.mean(yt < 0))))
