"""Predictor-in-the-loop curriculum climb (the closed-loop vision).

Two arms hill-climb the SAME model from the SAME init over a fixed pool of candidate
cohorts ("incoming data"), training cumulatively (weights persist across steps):

  RANDOM     : each step pick a random unused cohort  -> GRPO-train -> eval
  PREDICTOR  : each step score the unused cohorts by adv_std (the "disagree zone";
               a no-grad rollout = "a few tries"), train the BEST -> eval

We log held-out accuracy + NLL after every step (the climb) plus cumulative
generations (compute-fair x-axis). If the predictor arm climbs faster, the lift
predictor accelerates RL convergence -- the whole thesis.

Reuses the battery primitives verbatim (collect_trajectories = score, grpo_train =
train, eval_accuracy/eval_nll = the climb); the ONLY difference from the battery is
that the model is NOT reset between steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from tap import features as F
from tap.battery import (BatteryConfig, _load_policy, _snapshot, _reset, _question,
                         collect_trajectories, grpo_train, eval_accuracy, eval_nll, _append)
from tap.cohorts import random_cohorts
from tap.domains import get_domain


def _rows_of(cohort, rows_by_id):
    return [rows_by_id[p] for p in cohort.prompt_ids if p in rows_by_id]


def climb(model, tok, cfg: BatteryConfig, base_snap, pool, rows_by_id, probe_unigrams,
          eval_set, *, arm: str, seed: int, steps: int, cand_per_step: int,
          out_path: Path, baseline: tuple | None) -> None:
    """One arm's cumulative hill-climb; appends a log row per step to out_path."""
    import random
    import torch

    _reset(model, base_snap)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    rng = random.Random(1000 + seed)
    remaining = list(pool)
    gens = 0

    if baseline is not None:  # identical init for every arm -> reuse the step-0 eval
        acc, nll = baseline
    else:
        acc = eval_accuracy(model, tok, eval_set, cfg) if cfg.acc_eval else 0.0
        nll = eval_nll(model, tok, eval_set, cfg)
    _append(out_path, {"arm": arm, "seed": seed, "step": 0, "picked": None, "adv_std": None,
                       "acc": acc, "nll": nll, "gens": 0})
    print(f"[loop] {arm} s{seed} step0 acc={acc:.3f} nll={nll:.4f}", flush=True)

    for step in range(1, steps + 1):
        torch.manual_seed(7919 * seed + step)
        t0 = time.time()
        if arm == "predictor":
            cands = remaining if cand_per_step <= 0 else rng.sample(remaining, min(cand_per_step, len(remaining)))
            best = None  # (score, cohort, trajs, feats)
            for c in cands:
                rows = _rows_of(c, rows_by_id)
                if not rows:
                    continue
                noisy = set(map(str, c.meta.get("noisy_ids", [])))
                trajs, feats, _, _ = collect_trajectories(model, tok, rows, cfg, noisy, probe_unigrams)
                gens += cfg.group_size * len(rows)
                s = feats.get("adv_std")
                s = -1.0 if s is None else s
                if best is None or s > best[0]:
                    best = (s, c, trajs, feats)
            score, pick, pick_trajs, pick_feats = best
        else:  # random
            pick = rng.choice(remaining)
            rows = _rows_of(pick, rows_by_id)
            noisy = set(map(str, pick.meta.get("noisy_ids", [])))
            pick_trajs, pick_feats, _, _ = collect_trajectories(model, tok, rows, cfg, noisy, probe_unigrams)
            gens += cfg.group_size * len(rows)
            score = pick_feats.get("adv_std")

        ts = grpo_train(model, pick_trajs, cfg, opt, tok.pad_token_id)
        remaining.remove(pick)
        acc = eval_accuracy(model, tok, eval_set, cfg) if cfg.acc_eval else 0.0
        nll = eval_nll(model, tok, eval_set, cfg)
        _append(out_path, {"arm": arm, "seed": seed, "step": step, "picked": pick.name,
                           "adv_std": score, "frac_nondegenerate": pick_feats.get("frac_nondegenerate"),
                           "mean_reward": ts["mean_reward"], "kl_train": ts["kl"], "n_contrib": ts["n_contrib"],
                           "acc": acc, "nll": nll, "gens": gens, "remaining": len(remaining),
                           "wall_s": time.time() - t0})
        print(f"[loop] {arm} s{seed} step{step:02d} pick={pick.name:18s} adv_std={score:+.3f} "
              f"reward={ts['mean_reward']:.3f} acc={acc:.3f} nll={nll:.4f} gens={gens} "
              f"t={time.time() - t0:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/tap"))
    p.add_argument("--output", type=Path, default=Path("outputs/loop/loop.jsonl"))
    p.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--domain", default="compmath",
                   choices=("math", "code", "science", "mmlu", "compmath", "codemmlu"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--grpo-steps", type=int, default=8)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--eval-batch", type=int, default=0)
    p.add_argument("--gen-batch", type=int, default=48)
    p.add_argument("--probe-size", type=int, default=120, help="held-out eval set size (the climb metric)")
    p.add_argument("--cohort-size", type=int, default=4, help="problems per candidate batch")
    p.add_argument("--pool-size", type=int, default=40, help="number of candidate cohorts (incoming data)")
    p.add_argument("--steps", type=int, default=20, help="climb steps (cohorts trained on)")
    p.add_argument("--candidates-per-step", type=int, default=0, help="predictor scoring breadth; 0 => all remaining")
    p.add_argument("--random-seeds", type=int, default=3, help="random-arm repeats (it's noisy)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-acc-eval", dest="acc_eval", action="store_false")
    p.set_defaults(acc_eval=True)
    args = p.parse_args(argv)

    bundle = get_domain(args.domain).load_splits(args.data_dir, probe_size=args.probe_size,
                                                 fingerprint_size=16, seed=args.seed)
    rows_by_id = {r["id"]: r for r in bundle["train_rows"]}
    eval_set = bundle["probes"]["global"]
    pool = random_cohorts(bundle["train_rows"], n_cohorts=args.pool_size, size=args.cohort_size, seed=args.seed)
    if args.steps > len(pool):
        args.steps = len(pool)

    cfg = BatteryConfig(model_name=args.model_name, domain_name=args.domain, device=args.device, dtype=args.dtype,
                        grpo_steps=args.grpo_steps, group_size=args.group_size, max_new_tokens=args.max_new_tokens,
                        micro_batch=args.micro_batch, eval_batch=args.eval_batch, gen_batch=args.gen_batch,
                        acc_eval=args.acc_eval, lr=args.lr, seed=args.seed, temperature=args.temperature)

    model, tok = _load_policy(cfg)
    probe_unigrams = F._unigrams(tok(_question(r), add_special_tokens=False).input_ids for r in eval_set)
    base_snap = _snapshot(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # identical init -> compute the step-0 eval once and share it across arms
    acc0 = eval_accuracy(model, tok, eval_set, cfg) if cfg.acc_eval else 0.0
    nll0 = eval_nll(model, tok, eval_set, cfg)
    print(f"[loop] init acc={acc0:.3f} nll={nll0:.4f} | pool={len(pool)} steps={args.steps} "
          f"model={args.model_name} domain={args.domain}", flush=True)

    climb(model, tok, cfg, base_snap, pool, rows_by_id, probe_unigrams, eval_set,
          arm="predictor", seed=0, steps=args.steps, cand_per_step=args.candidates_per_step,
          out_path=args.output, baseline=(acc0, nll0))
    for s in range(args.random_seeds):
        climb(model, tok, cfg, base_snap, pool, rows_by_id, probe_unigrams, eval_set,
              arm="random", seed=s, steps=args.steps, cand_per_step=args.candidates_per_step,
              out_path=args.output, baseline=(acc0, nll0))
    print("[loop] done", flush=True)


if __name__ == "__main__":
    main()
