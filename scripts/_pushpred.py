#!/usr/bin/env python3
"""How high can the predictor's RANK go without more seeds?
  1. Noise ceiling (seed reliability -> max achievable spearman).
  2. Best single feature.
  3. Partial-corr residual: is ANY feature orthogonal to adv_std?
  4. Best multivariate model (ridge / gbdt) within-domain CV + transfer.
Target = NLL-lift, denoised per cohort (mean over seeds)."""
import json, numpy as np
from collections import defaultdict
from tap.metrics import spearman

rows = [json.loads(l) for l in open("outputs/enriched_labels.jsonl")]
DOMS = ["science", "codemmlu", "compmath", "mmlu"]

# candidate numeric features (features dict + a couple top-level)
FEX = ["adv_std", "adv_absmean", "frac_nondegenerate", "reward_std", "group_passrate_std",
       "group_passrate_mean", "pass_rate", "reward_mean", "frac_all_wrong", "frac_all_correct",
       "mean_logprob", "surprisal_mean", "entropy_mean", "logprob_p10", "logprob_p50",
       "logprob_p90", "confidence_slope", "redundancy_mean", "min_logprob_mean",
       "len_mean", "len_std", "len_max", "len_cv", "reward_gap", "distinct_frac"]
TOP = ["target_similarity", "step_frac"]

def fval(r, k):
    v = r.get(k) if k in TOP else (r.get("features") or {}).get(k)
    try:
        return float(v) if v is not None else np.nan
    except (TypeError, ValueError):
        return np.nan

# group by cohort (denoise seeds), keep per-seed targets for ceiling
g = defaultdict(list)
for r in rows:
    g[(r["domain"], r["candidate_id"])].append(r)

keys = list(g.keys())
dom = np.array([k[0] for k in keys])
Y = np.array([np.mean([r["lift_nll"] for r in g[k]]) for k in keys])
seedY = [[r["lift_nll"] for r in g[k]] for k in keys]
X = {}
for f in FEX + TOP:
    X[f] = np.array([np.nanmean([fval(r, f) for r in g[k]]) for k in keys], dtype=float)

def within(fv, y, mask):
    m = mask & np.isfinite(fv) & np.isfinite(y)
    return spearman(fv[m], y[m]) if m.sum() > 5 else float("nan")

# ---- 1. noise ceiling ---------------------------------------------------------
def sb(r1, k):  # spearman-brown to k measurements
    return k * r1 / (1 + (k - 1) * r1) if r1 > 0 else 0.0
print("== NOISE CEILING (max spearman a perfect predictor of the 3-seed mean could hit) ==")
print("%-10s %8s %8s %8s" % ("domain", "r1(seed)", "rel(k=3)", "ceiling"))
ceil = {}
for dm in DOMS:
    idx = [i for i, k in enumerate(keys) if k[0] == dm and len(seedY[i]) >= 3]
    a = np.array([seedY[i][0] for i in idx]); b = np.array([seedY[i][1] for i in idx]); c = np.array([seedY[i][2] for i in idx])
    r1 = np.mean([spearman(a, b), spearman(a, c), spearman(b, c)])
    rel = sb(r1, 3); cl = np.sqrt(max(rel, 0))
    ceil[dm] = cl
    print("%-10s %+8.3f %8.3f %8.3f" % (dm, r1, rel, cl))
print()

# ---- 2. best single features (within-domain) ----------------------------------
print("== TOP SINGLE FEATURES (within-domain spearman) ==")
sing = []
for f in FEX + TOP:
    per = {dm: within(X[f], Y, dom == dm) for dm in DOMS}
    avg = np.nanmean(list(per.values()))
    sing.append((avg, f, per))
sing.sort(reverse=True, key=lambda t: abs(t[0]))
print("%-20s %8s %8s %8s %8s %8s" % ("feature", *DOMS, "avg"))
for avg, f, per in sing[:8]:
    print("%-20s %+8.3f %+8.3f %+8.3f %+8.3f %+8.3f" % (f, *[per[d] for d in DOMS], avg))
best1 = sing[0][1]
print("ceiling avg = %.3f   best-single(%s) avg = %.3f" % (np.mean([ceil[d] for d in DOMS]), best1, sing[0][0]))
print()

# ---- 3. partial correlation: anything orthogonal to adv_std? ------------------
def rankz(x):
    o = np.full_like(x, np.nan, dtype=float); m = np.isfinite(x)
    r = np.argsort(np.argsort(x[m])).astype(float); o[m] = (r - r.mean()) / (r.std() + 1e-9); return o
print("== PARTIAL CORR(feature, NLL-lift | %s), within-domain (signal NOT in best feature) ==" % best1)
print("%-20s %8s %8s %8s %8s" % ("feature", *DOMS))
partrows = []
for f in FEX + TOP:
    if f == best1:
        continue
    cells = []
    for dm in DOMS:
        m = (dom == dm) & np.isfinite(X[f]) & np.isfinite(X[best1]) & np.isfinite(Y)
        if m.sum() < 8:
            cells.append(np.nan); continue
        ry, rf, rb = rankz(Y[m]), rankz(X[f][m]), rankz(X[best1][m])
        # residualize y and f on best, then corr
        by = ry - (np.dot(ry, rb) / np.dot(rb, rb)) * rb
        bf = rf - (np.dot(rf, rb) / np.dot(rb, rb)) * rb
        cells.append(float(np.corrcoef(by, bf)[0, 1]))
    partrows.append((np.nanmean(np.abs(cells)), f, cells))
partrows.sort(reverse=True)
for _, f, cells in partrows[:6]:
    print("%-20s %+8.3f %+8.3f %+8.3f %+8.3f" % (f, *cells))
print()

# ---- 4. multivariate models: within-domain CV + transfer ----------------------
def design(feats):
    M = np.column_stack([X[f] for f in feats])
    return M

def ridge_oof(M, y, groups, lam=1.0, folds=5):
    n = len(y); pred = np.full(n, np.nan)
    gid = np.array(groups)
    uniq = np.unique(gid); rng = np.random.default_rng(0); rng.shuffle(uniq)
    chunks = np.array_split(uniq, folds)
    for ch in chunks:
        te = np.isin(gid, ch); tr = ~te
        Xtr, Xte = M[tr], M[te]; ytr = y[tr]
        mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0) + 1e-9
        Ztr = np.nan_to_num((Xtr - mu) / sd); Zte = np.nan_to_num((Xte - mu) / sd)
        A = Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1])
        w = np.linalg.solve(A, Ztr.T @ (ytr - ytr.mean()))
        pred[te] = Zte @ w + ytr.mean()
    return pred

FSETS = {
    "best1 (adv_std)": [best1],
    "adv+frac+gpstd": ["adv_std", "frac_nondegenerate", "group_passrate_std"],
    "mech5": ["adv_std", "frac_nondegenerate", "reward_std", "group_passrate_std", "pass_rate"],
    "all25": FEX,
    "all+top": FEX + TOP,
}
print("== WITHIN-DOMAIN (ridge OOF, grouped by cohort) ==")
print("%-18s %8s %8s %8s %8s %8s" % ("featset", *DOMS, "avg"))
for nm, fs in FSETS.items():
    per = {}
    for dm in DOMS:
        m = dom == dm
        M = design(fs)[m]; y = Y[m]
        if len(fs) == 1:
            per[dm] = within(X[fs[0]], Y, m)
        else:
            p = ridge_oof(M, y, list(range(m.sum())), lam=3.0)
            per[dm] = spearman(p, y)
    print("%-18s %+8.3f %+8.3f %+8.3f %+8.3f %+8.3f" % (nm, *[per[d] for d in DOMS], np.nanmean(list(per.values()))))
print("ceiling             %+8.3f %+8.3f %+8.3f %+8.3f %+8.3f" % (*[ceil[d] for d in DOMS], np.mean([ceil[d] for d in DOMS])))
print()

print("== TRANSFER (leave-one-domain-out; train 3 -> rank held-out) ==")
print("%-18s %8s %8s %8s %8s %8s" % ("featset", *DOMS, "avg"))
for nm, fs in FSETS.items():
    per = {}
    for dm in DOMS:
        te = dom == dm; tr = ~te
        M = design(fs)
        if len(fs) == 1:
            per[dm] = within(X[fs[0]], Y, te)  # single feat: transfer == within rank
            continue
        Xtr, Xte = M[tr], M[te]; ytr = Y[tr]
        mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0) + 1e-9
        Ztr = np.nan_to_num((Xtr - mu) / sd); Zte = np.nan_to_num((Xte - mu) / sd)
        A = Ztr.T @ Ztr + 3.0 * np.eye(Ztr.shape[1])
        w = np.linalg.solve(A, Ztr.T @ (ytr - ytr.mean()))
        per[dm] = spearman(Zte @ w + ytr.mean(), Y[te])
    print("%-18s %+8.3f %+8.3f %+8.3f %+8.3f %+8.3f" % (nm, *[per[d] for d in DOMS], np.nanmean(list(per.values()))))
