#!/usr/bin/env python3
"""Project predictor rank vs #seeds.
Two things improve with k seeds: (1) the LABEL noise ceiling = sqrt(reliability_k),
(2) the FEATURE (adv_std) also denoises. Our predictor corr(feat_k, label_k) is
attenuated by both. Estimate the disattenuated true corr (rho_true = the asymptote
even with infinite seeds), then project pred(k) = rho_true*sqrt(rel_feat_k*rel_lab_k)."""
import json, numpy as np
from collections import defaultdict
from itertools import combinations
from tap.metrics import spearman

rows = [json.loads(l) for l in open("outputs/enriched_labels.jsonl")]
DOMS = ["science", "codemmlu", "compmath", "mmlu"]

def adv(r):
    v = (r.get("features") or {}).get("adv_std")
    return float(v) if v is not None else np.nan

g = defaultdict(list)
for r in rows:
    g[(r["domain"], r["candidate_id"])].append(r)

def sb(r1, k):  # spearman-brown reliability of a k-seed mean
    r1 = max(r1, 1e-6)
    return k * r1 / (1 + (k - 1) * r1)

def pairwise_r1(seedvecs):  # avg pairwise spearman across the 3 seed slots
    a, b, c = seedvecs
    return float(np.mean([spearman(a, b), spearman(a, c), spearman(b, c)]))

KS = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
emp = {dm: {} for dm in DOMS}
proj = {dm: {} for dm in DOMS}
ceil = {dm: {} for dm in DOMS}
rho = {}

for dm in DOMS:
    coh = [v for (d, c), v in g.items() if d == dm and len(v) >= 3]
    coh = [sorted(v, key=lambda r: r.get("seed", 0))[:3] for v in coh]
    nll = [[r["lift_nll"] for r in v] for v in coh]
    ad = [[adv(r) for r in v] for v in coh]
    nllT = list(zip(*nll)); adT = list(zip(*ad))  # 3 slots x cohorts
    nllT = [np.array(x) for x in nllT]; adT = [np.array(x) for x in adT]
    r1_lab = pairwise_r1(nllT)
    r1_fea = pairwise_r1(adT)
    # empirical predictor corr at k=1,2,3 (avg over seed combinations)
    for k in (1, 2, 3):
        cs = []
        for idx in combinations(range(3), k):
            fk = np.mean([adT[i] for i in idx], axis=0)
            yk = np.mean([nllT[i] for i in idx], axis=0)
            m = np.isfinite(fk) & np.isfinite(yk)
            cs.append(spearman(fk[m], yk[m]))
        emp[dm][k] = float(np.mean(cs))
    # disattenuate at k=3 -> true feature<->truth corr (the real asymptote)
    rho_true = emp[dm][3] / np.sqrt(sb(r1_fea, 3) * sb(r1_lab, 3))
    rho_true = min(rho_true, 1.0)
    rho[dm] = rho_true
    for k in KS:
        ceil[dm][k] = np.sqrt(sb(r1_lab, k))
        proj[dm][k] = rho_true * np.sqrt(sb(r1_fea, k) * sb(r1_lab, k))

def avg(d, k):
    return float(np.mean([d[dm][k] for dm in DOMS]))

print("per-domain single-seed reliability r1 (label) & disattenuated asymptote rho_true:")
print("%-10s %10s %12s" % ("domain", "r1(label)", "rho_true"))
for dm in DOMS:
    coh = [v for (d, c), v in g.items() if d == dm and len(v) >= 3]
    print("%-10s %10.3f %12.3f" % (dm, pairwise_r1([np.array(x) for x in zip(*[[r['lift_nll'] for r in sorted(v,key=lambda r:r.get('seed',0))[:3]] for v in coh])]), rho[dm]))

print("\nEMPIRICAL predictor corr (validation that denoising works):")
print("%-10s %7s %7s %7s" % ("domain", "k=1", "k=2", "k=3"))
for dm in DOMS:
    print("%-10s %7.3f %7.3f %7.3f" % (dm, emp[dm][1], emp[dm][2], emp[dm][3]))
print("%-10s %7.3f %7.3f %7.3f" % ("AVG", avg(emp, 1), avg(emp, 2), avg(emp, 3)))

print("\nPROJECTED predictor rank by #seeds (and the label ceiling):")
hdr = "  ".join("k=%-2d" % k for k in KS)
print("%-12s %s" % ("", hdr))
for dm in DOMS:
    print("%-12s %s" % (dm, "  ".join("%4.2f" % proj[dm][k] for k in KS)))
print("%-12s %s" % ("AVG(pred)", "  ".join("%4.2f" % avg(proj, k) for k in KS)))
print("%-12s %s" % ("AVG(ceil)", "  ".join("%4.2f" % avg(ceil, k) for k in KS)))
