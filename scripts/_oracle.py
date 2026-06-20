#!/usr/bin/env python3
"""Noise-free oracle via seed cross-fitting.
The naive oracle ranks cohorts by their measured 3-seed lift and is re-scored on
the SAME labels -> it chases noise and is inflated. Honest version: SELECT on a
subset of seeds, EVALUATE on a held-out seed (independent noise) so noise-chasing
can't pay off. Also a parametric TRUE oracle (disattenuated variance -> infinite
seeds). Then % of oracle gap captured is computed against an honest ceiling."""
import json, numpy as np
from collections import defaultdict
from tap.metrics import spearman

rows = [json.loads(l) for l in open("outputs/enriched_labels.jsonl")]
DOMS = ["science", "codemmlu", "compmath", "mmlu"]
EZ = {0.10: 1.7550, 0.20: 1.3998, 0.30: 1.1590}  # E[mean of top-p of standard normal]

def adv(r):
    v = (r.get("features") or {}).get("adv_std")
    return float(v) if v is not None else np.nan

g = defaultdict(list)
for r in rows:
    g[(r["domain"], r["candidate_id"])].append(r)

# per domain: arrays [cohort, seed] for lift and adv (cohorts with exactly 3 seeds)
def domain_arrays(dmn):
    L, A = [], []
    for (d, c), v in g.items():
        if d != dmn or len(v) < 3:
            continue
        v = sorted(v, key=lambda r: r.get("seed", 0))[:3]
        L.append([r["lift_nll"] for r in v]); A.append([adv(r) for r in v])
    return np.array(L, float), np.array(A, float)

def topk_mean(score, value, frac):
    k = max(int(round(len(value) * frac)), 1)
    return value[np.argsort(-score)[:k]].mean()

def sb(r1, k):
    r1 = max(r1, 1e-6); return k * r1 / (1 + (k - 1) * r1)

def r1_label(L):
    a, b, c = L[:, 0], L[:, 1], L[:, 2]
    return float(np.mean([spearman(a, b), spearman(a, c), spearman(b, c)]))

for frac in (0.10, 0.20, 0.30):
    res = defaultdict(list)
    for dmn in DOMS:
        L, A = domain_arrays(dmn)
        mean = L.mean()
        # --- naive (noise-inflated): select & eval on the same 3-seed mean ---
        m3 = L.mean(1)
        naive = topk_mean(m3, m3, frac)
        # --- honest cross-fit: select on 2 seeds, eval on held-out seed (rotate) ---
        hp, ho = [], []
        for h in range(3):
            sel = [s for s in range(3) if s != h]
            lift_sel = L[:, sel].mean(1); adv_sel = A[:, sel].mean(1); ev = L[:, h]
            ho.append(topk_mean(lift_sel, ev, frac))   # honest oracle
            hp.append(topk_mean(adv_sel, ev, frac))    # predictor, eval on held-out
        honest_oracle = np.mean(ho); pred_eval = np.mean(hp)
        # --- parametric TRUE oracle (infinite seeds): disattenuate variance ---
        rel3 = sb(r1_label(L), 3)
        sd_true = np.sqrt(max(rel3, 0)) * m3.std()
        true_oracle = mean + EZ[frac] * sd_true
        res["random"].append(mean); res["naive"].append(naive)
        res["honest_oracle"].append(honest_oracle); res["true_oracle"].append(true_oracle)
        res["pred"].append(pred_eval)
    R = {k: np.mean(v) for k, v in res.items()}
    print("=== TOP-%d%% (avg over domains) ===" % int(frac * 100))
    print("  random (no skill):           %.3f" % R["random"])
    print("  predictor (adv_std):         %.3f" % R["pred"])
    print("  naive oracle  (noise-chase): %.3f   <- inflated" % R["naive"])
    print("  honest oracle (cross-fit):   %.3f   <- noise can't be chased" % R["honest_oracle"])
    print("  true oracle   (infinite k):  %.3f   <- parametric, noise-free" % R["true_oracle"])
    def cap(o):
        return 100 * (R["pred"] - R["random"]) / (o - R["random"]) if o > R["random"] else float("nan")
    print("  %% of gap captured  vs naive=%.0f%%  vs honest=%.0f%%  vs true=%.0f%%" %
          (cap(R["naive"]), cap(R["honest_oracle"]), cap(R["true_oracle"])))
    print("  lift over random: predictor=+%.0f%%, honest-oracle=+%.0f%%" %
          (100 * (R["pred"] / R["random"] - 1), 100 * (R["honest_oracle"] / R["random"] - 1)))
    print()
