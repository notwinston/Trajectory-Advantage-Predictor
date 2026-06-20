#!/usr/bin/env python3
"""What distinguishes high-NLL-gain cohorts from low? Rank cohorts WITHIN each
domain (removes domain-mean confound), pool top-third vs bottom-third, compare
feature means (Cohen's d) + cohort-type mix."""
import glob, json
from collections import defaultdict, Counter
import numpy as np
from tap.predictor import build_xy

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
kinds = [(r.get("cohort") or {}).get("kind", "?") for r in rows]
g = defaultdict(list)
for i, (dm, c) in enumerate(zip(doms, cids)):
    g[(dm, c)].append(i)
Xc = np.array([np.nanmean(X[ix], axis=0) for ix in g.values()])
yc = np.array([float(np.mean(y[ix])) for ix in g.values()])
domc = np.array([k[0] for k in g.keys()])
kindc = [kinds[ix[0]] for ix in g.values()]

# within-domain percentile rank of NLL-lift
rank = np.zeros(len(yc))
for dm in DIRS:
    m = np.where(domc == dm)[0]
    order = np.argsort(yc[m])
    pr = np.empty(len(m)); pr[order] = np.linspace(0, 1, len(m))
    rank[m] = pr
hi = rank >= 0.667
lo = rank <= 0.333
print("HIGH-NLL-gain cohorts: %d   LOW: %d  (within-domain terciles)\n" % (hi.sum(), lo.sum()))

rowsdiff = []
for j, nm in enumerate(names):
    a, b = Xc[hi, j], Xc[lo, j]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        continue
    sd = np.sqrt((np.var(a) + np.var(b)) / 2) or 1e-9
    dcoh = (np.mean(a) - np.mean(b)) / sd
    rowsdiff.append((nm, np.mean(a), np.mean(b), dcoh))
rowsdiff.sort(key=lambda r: -abs(r[3]))
print("%-22s %9s %9s %8s" % ("feature", "HIGH", "LOW", "cohen_d"))
for nm, mh, ml, dc in rowsdiff[:12]:
    print("%-22s %+9.3f %+9.3f %+8.2f" % (nm, mh, ml, dc))

print("\ncohort-type mix (fraction of HIGH vs LOW):")
ch = Counter(k for k, h in zip(kindc, hi) if h); cl = Counter(k for k, l in zip(kindc, lo) if l)
for k in sorted(set(ch) | set(cl)):
    print("  %-14s HIGH=%.2f  LOW=%.2f" % (k, ch[k] / max(hi.sum(), 1), cl[k] / max(lo.sum(), 1)))
