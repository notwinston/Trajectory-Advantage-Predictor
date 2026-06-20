#!/usr/bin/env python3
"""Engineer NEW cheap features from the persisted raw rollouts and correlate
each with NLL-lift (per domain + pooled). Tests whether genuinely-new signals
(length dist, logprob trajectory, reward gap, diversity, repetition) carry
information beyond the existing summary features."""
import glob, json
from collections import defaultdict, Counter
import numpy as np
from tap.metrics import spearman

DIRS = {"science": "outputs/q_science_8x", "codemmlu": "outputs/q_codemmlu",
        "compmath": "outputs/q_compmath", "mmlu": "outputs/q_mmlu"}

def keyf(r):
    return f"{r.get('chain_id')}.{r.get('anchor_index')}.{r.get('candidate_id')}.{r.get('seed')}"

def feats(trajs):
    lens = [len(t["comp_ids"]) for t in trajs if t.get("comp_ids")]
    allt = [x for t in trajs for x in (t.get("token_logp") or [])]
    rew = [t.get("reward", 0.0) for t in trajs]
    adv = [t.get("advantage", 0.0) for t in trajs]
    f = {}
    if lens:
        f["len_std"] = float(np.std(lens)); f["len_cv"] = float(np.std(lens) / (np.mean(lens) + 1e-9))
        f["len_max"] = float(max(lens)); f["frac_long"] = float(np.mean([l > 230 for l in lens]))
    if allt:
        f["lp_std"] = float(np.std(allt)); f["lp_min"] = float(np.min(allt)); f["ppl"] = float(np.exp(-np.mean(allt)))
    slopes = []
    for t in trajs:
        tlp = t.get("token_logp") or []
        if len(tlp) >= 4:
            h = len(tlp) // 2; slopes.append(np.mean(tlp[h:]) - np.mean(tlp[:h]))
    if slopes:
        f["lp_slope"] = float(np.mean(slopes))
    if rew:
        f["reward_gap"] = float(max(rew) - min(rew))
    if adv:
        f["adv_absmean"] = float(np.mean(np.abs(adv))); f["adv_std"] = float(np.std(adv))
    comps = [tuple(t["comp_ids"]) for t in trajs if t.get("comp_ids")]
    if comps:
        f["distinct_frac"] = len(set(comps)) / len(comps)
    reps = []
    for t in trajs:
        ci = t.get("comp_ids") or []
        if ci:
            reps.append(max(Counter(ci).values()) / len(ci))
    if reps:
        f["max_tok_rep"] = float(np.mean(reps))
    return f

# load labels (NLL-lift) keyed
labkey = {}
for dom, d in DIRS.items():
    for fn in glob.glob(d + "/labels_shard_*.jsonl"):
        if "rollouts" in fn:
            continue
        for l in open(fn):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("lift_nll") is None:
                continue
            labkey[(dom, keyf(r))] = (r.get("candidate_id"), r.get("lift_nll"))

# load rollouts, compute feats, join to labels
recs = []
for dom, d in DIRS.items():
    for fn in glob.glob(d + "/labels_shard_*_rollouts.jsonl"):
        for l in open(fn):
            try:
                ro = json.loads(l)
            except Exception:
                continue
            kk = (dom, ro.get("key"))
            if kk not in labkey:
                continue
            cid, nll = labkey[kk]
            ff = feats(ro.get("trajs") or [])
            ff["_dom"] = dom; ff["_cid"] = cid; ff["_nll"] = nll
            recs.append(ff)

print("joined rollout+label records: %d" % len(recs))
featnames = sorted({k for r in recs for k in r if not k.startswith("_")})
# denoise per (dom,cid)
g = defaultdict(list)
for r in recs:
    g[(r["_dom"], r["_cid"])].append(r)
doms = [k[0] for k in g]
Y = np.array([np.mean([r["_nll"] for r in v]) for v in g.values()])
FX = {fn: np.array([np.nanmean([r.get(fn, np.nan) for r in v]) for v in g.values()]) for fn in featnames}
doms = np.array(doms)

print("\nspearman(NEW feature, NLL-lift):   [baseline frac_nondegenerate pooled = 0.77]")
print("%-14s %8s %8s %8s %8s %8s" % ("feature", "science", "codemmlu", "compmath", "mmlu", "POOLED"))
ranked = []
for fn in featnames:
    fv = FX[fn]
    pooled = spearman(fv[np.isfinite(fv)], Y[np.isfinite(fv)])
    ranked.append((abs(pooled), fn, fv, pooled))
ranked.sort(reverse=True)
for _, fn, fv, pooled in ranked:
    cells = []
    for dm in ["science", "codemmlu", "compmath", "mmlu"]:
        m = (doms == dm) & np.isfinite(fv)
        cells.append("%+.3f" % spearman(fv[m], Y[m]) if m.sum() > 5 else "  -  ")
    print("%-14s %8s %8s %8s %8s %+8.3f" % (fn, *cells, pooled))
