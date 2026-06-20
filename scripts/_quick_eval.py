#!/usr/bin/env python3
"""Quick pooled train/test eval of the lift predictor on collected labels.

Random 80/20 split BY COHORT (all seeds of a cohort on the same side => no
seed leakage). Trains the GBDT on train cohorts, reports test metrics + a
mean baseline. Preview only; the rigorous test is leave-one-domain-out later.
"""
import glob, json, random
from collections import Counter

import numpy as np
from tap.predictor import build_xy, LiftPredictor, _all_metrics

DIRS = {
    "science":  "outputs/q_science_8x",
    "codemmlu": "outputs/q_codemmlu",
    "compmath": "outputs/q_compmath",
    "mmlu":     "outputs/ckpt/31645410",   # still running -> use latest checkpoint
}

seen, rows = set(), []
for dom, d in DIRS.items():
    for f in glob.glob(f"{d}/labels_shard_*.jsonl"):
        if "rollouts" in f:
            continue
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            key = (r.get("domain", dom), r.get("chain_id"), r.get("anchor_index"),
                   r.get("candidate_id"), r.get("seed"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)

print(f"merged {len(rows)} labels | per-domain: {dict(Counter(r.get('domain') for r in rows))}")

# group by cohort (domain + candidate_id) so all 3 seeds stay on one side
cohorts: dict = {}
for r in rows:
    cohorts.setdefault((r.get("domain"), r.get("candidate_id")), []).append(r)
ck = list(cohorts)
random.Random(0).shuffle(ck)
n_test = max(1, int(0.20 * len(ck)))
test_rows = [r for c in ck[:n_test] for r in cohorts[c]]
train_rows = [r for c in ck[n_test:] for r in cohorts[c]]
print(f"cohorts={len(ck)} -> train={len(ck)-n_test} cohorts/{len(train_rows)} rows, "
      f"test={n_test} cohorts/{len(test_rows)} rows")

a = build_xy(train_rows, with_context=True, label="acc")
b = build_xy(test_rows, with_context=True, label="acc")
model = LiftPredictor(backend="auto", monotone=True).fit(a["X"], a["y"], names=a["names"])
pred = model.predict(b["X"])

m = _all_metrics(pred, b["y"], b["anchors"], 0.0)
base = _all_metrics(np.full(len(b["y"]), float(np.mean(a["y"]))), b["y"], b["anchors"], 0.0)
keys = ["pearson", "spearman", "sign_accuracy", "within_anchor_spearman",
        "pairwise_ranking_accuracy", "rmse"]
print("\n=== TEST METRICS (GBDT vs mean-baseline) ===")
for k in keys:
    print(f"  {k:28s} GBDT={float(m[k]):+.4f}   baseline={float(base[k]):+.4f}")
print(f"  backend={model._resolved}  n_features={a['X'].shape[1]}")

print("\n=== sample predictions (test) ===")
order = np.argsort(-np.abs(b["y"]))[:8]
for i in order:
    r = test_rows[i]
    print(f"  {r.get('domain'):9s} {str(r.get('candidate_id'))[:22]:22s} "
          f"pred={pred[i]:+.3f}  actual={b['y'][i]:+.3f}")
