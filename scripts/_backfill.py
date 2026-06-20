#!/usr/bin/env python3
"""Backfill the NEW cheap features onto the existing labels by re-aggregating the
persisted raw rollouts through the SAME summarize_rollouts code path (consistency).
Writes a complete, analysis-ready dataset to outputs/enriched_labels.jsonl --
originals untouched. Future labels carry these natively (battery already updated)."""
import glob, json, os
from tap.features import summarize_rollouts

DIRS = {"science": "outputs/q_science_8x", "codemmlu": "outputs/q_codemmlu",
        "compmath": "outputs/q_compmath", "mmlu": "outputs/q_mmlu"}

def keyf(r):
    return f"{r.get('chain_id')}.{r.get('anchor_index')}.{r.get('candidate_id')}.{r.get('seed')}"

def trajs_to_dicts(trajs):
    out = []
    for t in trajs:
        tlp = t.get("token_logp") or []
        ci = t.get("comp_ids") or []
        half = max(len(tlp) // 2, 1)
        out.append({
            "group_id": t.get("group_id"), "reward": t.get("reward"), "advantage": t.get("advantage"),
            "completion_tokens": len(ci),
            "mean_logprob": (sum(tlp) / len(tlp)) if tlp else None,
            "early_logprob": (sum(tlp[:half]) / half) if tlp else None,
            "late_logprob": (sum(tlp[half:]) / max(len(tlp) - half, 1)) if tlp else None,
            "min_logprob": (min(tlp) if tlp else None),
            "comp_hash": hash(tuple(ci))})
    return out

roll = {}
for dom, d in DIRS.items():
    for fn in glob.glob(d + "/labels_shard_*_rollouts.jsonl"):
        for l in open(fn):
            try:
                ro = json.loads(l)
            except Exception:
                continue
            roll[(dom, ro.get("key"))] = ro.get("trajs") or []

os.makedirs("outputs", exist_ok=True)
n = matched = 0
keep = ("domain", "candidate_id", "seed", "chain_id", "anchor_index", "cohort",
        "lift_nll", "lift_acc", "nll_before", "nll_after", "kl_drift", "step_frac",
        "target_similarity", "rollout_count")
with open("outputs/enriched_labels.jsonl", "w") as out:
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
                k = (dom, keyf(r))
                if k in roll:
                    feats = summarize_rollouts(trajs_to_dicts(roll[k])).as_dict()
                    matched += 1
                else:
                    feats = r.get("reward_summary") or {}
                rec = {kk: r.get(kk) for kk in keep}
                rec["domain"] = dom
                rec["features"] = feats
                out.write(json.dumps(rec, default=float) + "\n")
                n += 1
print(f"wrote {n} enriched labels ({matched} re-aggregated from rollouts) -> outputs/enriched_labels.jsonl")
# show the new fields are present + populated
import statistics
ex = [json.loads(l) for l in open("outputs/enriched_labels.jsonl")]
new = ["adv_std", "adv_absmean", "len_max", "len_cv", "reward_gap", "distinct_frac", "min_logprob_mean"]
print("new fields populated (non-null count / %d):" % len(ex))
for kf in new:
    vals = [e["features"].get(kf) for e in ex if e["features"].get(kf) is not None]
    print("  %-18s %4d  mean=%+.3f" % (kf, len(vals), statistics.fmean(vals) if vals else float("nan")))
