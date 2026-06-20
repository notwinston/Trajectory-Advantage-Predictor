#!/usr/bin/env python3
"""Full prediction-quality panel for NLL-lift (predictor = LOO linear on adv_std):
  rank:      spearman, kendall-tau
  magnitude: pearson, out-of-sample R2, RMSE, MAE (vs target std)
  decision:  selection lift -- pick top-k% by predictor, % of oracle gap captured."""
import json, numpy as np
from collections import defaultdict
from tap.metrics import spearman

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
    n = len(y); p = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        xs, ys = x[m], y[m]
        b1 = np.cov(xs, ys, bias=True)[0, 1] / (np.var(xs) + 1e-12)
        p[i] = (ys.mean() - b1 * xs.mean()) + b1 * x[i]
    return p

def kendall(a, b):
    n = len(a); c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0: c += 1
            elif s < 0: d += 1
    return (c - d) / (0.5 * n * (n - 1)) if n > 1 else float("nan")

def r2_oos(yp, yt):
    return 1 - np.sum((yt - yp) ** 2) / (np.sum((yt - yt.mean()) ** 2) + 1e-12)

def selection(pred, actual, frac):
    k = max(int(round(len(actual) * frac)), 1)
    order = np.argsort(-pred)
    pick = actual[order[:k]].mean()
    oracle = np.sort(actual)[::-1][:k].mean()
    rand = actual.mean()
    cap = (pick - rand) / (oracle - rand) if oracle > rand else float("nan")
    return rand, pick, oracle, cap

print("RANK + MAGNITUDE quality (per-domain, LOO linear on adv_std):")
print("%-9s %4s %8s %8s %8s %8s %7s %7s %8s" %
      ("domain", "n", "spear", "kendall", "pearson", "R2(oos)", "RMSE", "MAE", "tgt_std"))
PY, PT = [], []
for dmn in DOMS:
    m = (dom == dmn) & np.isfinite(A) & np.isfinite(Y)
    x, y = A[m], Y[m]
    yp = loo_pred(x, y)
    PY.append(yp); PT.append(y)
    pear = np.corrcoef(yp, y)[0, 1]
    rmse = np.sqrt(np.mean((yp - y) ** 2)); mae = np.mean(np.abs(yp - y))
    print("%-9s %4d %8.3f %8.3f %8.3f %8.3f %7.3f %7.3f %8.3f" %
          (dmn, len(y), spearman(x, y), kendall(x, y), pear, r2_oos(yp, y), rmse, mae, y.std()))
yp = np.concatenate(PY); yt = np.concatenate(PT)
print("%-9s %4d %8.3f %8s %8.3f %8.3f %7.3f %7.3f %8.3f" %
      ("POOLED", len(yt), spearman(yp, yt), "-", np.corrcoef(yp, yt)[0, 1], r2_oos(yp, yt),
       np.sqrt(np.mean((yp - yt) ** 2)), np.mean(np.abs(yp - yt)), yt.std()))

print("\nDECISION quality -- pick top cohorts by predictor (per-domain, then avg):")
for frac in (0.10, 0.20, 0.30):
    caps, picks, rands, orcs = [], [], [], []
    for dmn in DOMS:
        m = (dom == dmn) & np.isfinite(A) & np.isfinite(Y)
        x, y = A[m], Y[m]
        yp = loo_pred(x, y)
        rand, pick, oracle, cap = selection(yp, y, frac)
        caps.append(cap); picks.append(pick); rands.append(rand); orcs.append(oracle)
    print("  top-%2d%%: picked-lift %.3f  vs random %.3f  vs oracle %.3f  -> %.0f%% of oracle gap captured" %
          (int(frac * 100), np.mean(picks), np.mean(rands), np.mean(orcs), 100 * np.nanmean(caps)))

print("\nnote: ~31%% of cohorts have actual lift==0 (degenerate); they are part of the problem")
print("      (a good predictor should rank them low) and are included above.")
