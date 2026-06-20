"""Predictor-in-the-loop curriculum climb with an ONLINE-LEARNING predictor.

Two arms hill-climb the SAME model from the SAME init over a pool of candidate
cohorts ("incoming data"), training cumulatively (weights persist across steps):

  RANDOM  : each step pick a random unused cohort -> GRPO-train -> eval
  ONLINE  : each step DYNAMICALLY score the unused cohorts (re-roll with the CURRENT
            model = "a few tries"), pick the highest PREDICTED lift, train it, then
            observe the REALIZED lift (drop in held-out NLL) and REFIT the predictor
            on (features -> realized lift). The predictor starts as the adv_std
            heuristic and learns online -- the "it keeps learning" loop.

The selected cohort's scoring rollout is on-policy (current model), so it's reused
for training (no extra rollout). NLL is evaluated every step (cheap, teacher-forced
= the dense climb signal + the predictor's target); greedy accuracy every
--acc-every steps. Logs predicted-vs-realized lift so we can see the predictor
sharpen over time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

from tap import features as F
from tap.battery import (BatteryConfig, _load_policy, _snapshot, _reset, _question,
                         collect_trajectories, grpo_train, eval_accuracy, eval_nll, _append)
from tap.cohorts import random_cohorts
from tap.domains import get_domain


class OnlinePredictor:
    """Tiny online ridge over rollout features; cold-starts as the adv_std heuristic,
    then refits on (features -> realized lift) as the loop observes real gains."""

    KEYS = ("adv_std", "frac_nondegenerate", "reward_std", "group_passrate_std",
            "mean_logprob", "len_mean")

    def __init__(self, lam: float = 1.0, warmup: int = 4):
        self.X: list[list[float]] = []
        self.y: list[float] = []
        self.lam = lam
        self.warmup = warmup
        self._w = None
        self._mu = self._sd = None
        self._ybar = 0.0

    def vec(self, feats: dict) -> list[float]:
        return [float(feats.get(k)) if feats.get(k) is not None else 0.0 for k in self.KEYS]

    def observe(self, feats: dict, lift: float) -> None:
        self.X.append(self.vec(feats))
        self.y.append(float(lift))
        self._refit()

    def _refit(self) -> None:
        if len(self.y) < self.warmup:
            self._w = None
            return
        import numpy as np
        X = np.array(self.X); y = np.array(self.y)
        self._mu = X.mean(0); self._sd = X.std(0) + 1e-9
        Z = (X - self._mu) / self._sd
        self._ybar = float(y.mean())
        self._w = np.linalg.solve(Z.T @ Z + self.lam * np.eye(Z.shape[1]), Z.T @ (y - self._ybar))

    def predict(self, feats: dict) -> float:
        if self._w is None:  # cold start: rank by adv_std (the known-good heuristic)
            v = feats.get("adv_std")
            return float(v) if v is not None else 0.0
        import numpy as np
        z = (np.array(self.vec(feats)) - self._mu) / self._sd
        return float(z @ self._w + self._ybar)

    @property
    def learned(self) -> bool:
        return self._w is not None


def _rows_of(cohort, rows_by_id):
    return [rows_by_id[p] for p in cohort.prompt_ids if p in rows_by_id]


def _roll(model, tok, cohort, rows_by_id, cfg, probe_unigrams):
    rows = _rows_of(cohort, rows_by_id)
    noisy = set(map(str, cohort.meta.get("noisy_ids", [])))
    trajs, feats, _, _ = collect_trajectories(model, tok, rows, cfg, noisy, probe_unigrams)
    return trajs, feats, cfg.group_size * len(rows)


def climb(model, tok, cfg, base_snap, pool, rows_by_id, probe_unigrams, eval_set, *,
          arm, seed, steps, cand_per_step, acc_every, out_path, nll0, acc0,
          score_mode="cached", feat_cache=None):
    import torch

    _reset(model, base_snap)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    rng = random.Random(1000 + seed)
    predictor = OnlinePredictor() if arm == "online" else None
    remaining = list(pool)
    gens = 0
    prev_nll = nll0

    _append(out_path, {"arm": arm, "seed": seed, "step": 0, "picked": None, "acc": acc0,
                       "nll": nll0, "gens": 0, "predicted_lift": None, "realized_lift": None})
    print(f"[loop] {arm} s{seed} step0 acc={acc0:.3f} nll={nll0:.4f}", flush=True)

    for step in range(1, steps + 1):
        torch.manual_seed(7919 * (seed + 1) + step)
        t0 = time.time()
        if arm == "online" and score_mode == "dynamic":   # re-score (re-roll) candidates each step
            cands = remaining if cand_per_step <= 0 else rng.sample(remaining, min(cand_per_step, len(remaining)))
            best = None
            for c in cands:
                if not _rows_of(c, rows_by_id):
                    continue
                trajs, feats, g = _roll(model, tok, c, rows_by_id, cfg, probe_unigrams)
                gens += g
                pv = predictor.predict(feats)
                if best is None or pv > best[0]:
                    best = (pv, c, trajs, feats)
            predicted_lift, pick, pick_trajs, score_feats = best
        elif arm == "online":                              # cached: rank by one-time measured features
            predicted_lift, pick = max(((predictor.predict(feat_cache[c.name]), c)
                                        for c in remaining if c.name in feat_cache), key=lambda x: x[0])
            score_feats = feat_cache[pick.name]
            pick_trajs, _, g = _roll(model, tok, pick, rows_by_id, cfg, probe_unigrams)  # on-policy train rollout
            gens += g
        else:                                              # random
            pick = rng.choice(remaining)
            pick_trajs, score_feats, g = _roll(model, tok, pick, rows_by_id, cfg, probe_unigrams)
            gens += g
            predicted_lift = None

        ts = grpo_train(model, pick_trajs, cfg, opt, tok.pad_token_id)
        remaining.remove(pick)
        nll = eval_nll(model, tok, eval_set, cfg)
        realized_lift = prev_nll - nll                                    # the REAL gain
        do_acc = cfg.acc_eval and (step % acc_every == 0 or step == steps)
        acc = eval_accuracy(model, tok, eval_set, cfg) if do_acc else None
        if predictor is not None:
            predictor.observe(score_feats, realized_lift)                # <-- predictor LEARNS
        pick_feats = score_feats

        _append(out_path, {"arm": arm, "seed": seed, "step": step, "picked": pick.name,
                           "adv_std": pick_feats.get("adv_std"), "predicted_lift": predicted_lift,
                           "realized_lift": realized_lift, "predictor_learned": bool(predictor and predictor.learned),
                           "mean_reward": ts["mean_reward"], "acc": acc, "nll": nll, "gens": gens,
                           "remaining": len(remaining), "wall_s": time.time() - t0})
        ap = "  acc=%.3f" % acc if acc is not None else ""
        pl = "" if predicted_lift is None else " pred=%+.3f" % predicted_lift
        print(f"[loop] {arm} s{seed} step{step:02d} pick={pick.name:16s}{pl} real={realized_lift:+.3f} "
              f"nll={nll:.4f}{ap} gens={gens} t={time.time()-t0:.0f}s", flush=True)
        prev_nll = nll


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
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--eval-batch", type=int, default=0)
    p.add_argument("--gen-batch", type=int, default=64)
    p.add_argument("--probe-size", type=int, default=64, help="held-out eval set size")
    p.add_argument("--cohort-size", type=int, default=4)
    p.add_argument("--pool-size", type=int, default=30)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--candidates-per-step", type=int, default=8, help="dynamic scoring breadth; 0=all remaining")
    p.add_argument("--score-mode", choices=("cached", "dynamic"), default="cached",
                   help="cached: measure cohorts ONCE then rank (fast); dynamic: re-roll candidates each step (faithful, ~3x slower)")
    p.add_argument("--acc-every", type=int, default=3, help="greedy accuracy eval cadence (nll is every step)")
    p.add_argument("--random-seeds", type=int, default=2)
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
    acc0 = eval_accuracy(model, tok, eval_set, cfg) if cfg.acc_eval else 0.0
    nll0 = eval_nll(model, tok, eval_set, cfg)
    print(f"[loop] init acc={acc0:.3f} nll={nll0:.4f} | pool={len(pool)} steps={args.steps} "
          f"score={args.score_mode} model={args.model_name} domain={args.domain}", flush=True)

    feat_cache = None
    if args.score_mode == "cached":   # measure every cohort ONCE (like the battery), then rank from it
        feat_cache = {}
        for i, c in enumerate(pool):
            _, feats, _ = _roll(model, tok, c, rows_by_id, cfg, probe_unigrams)
            feat_cache[c.name] = feats
        print(f"[loop] cached features for {len(feat_cache)} cohorts (one-time measurement)", flush=True)

    kw = dict(steps=args.steps, cand_per_step=args.candidates_per_step, acc_every=args.acc_every,
              out_path=args.output, nll0=nll0, acc0=acc0, score_mode=args.score_mode, feat_cache=feat_cache)
    climb(model, tok, cfg, base_snap, pool, rows_by_id, probe_unigrams, eval_set, arm="online", seed=0, **kw)
    for s in range(args.random_seeds):
        climb(model, tok, cfg, base_snap, pool, rows_by_id, probe_unigrams, eval_set, arm="random", seed=s, **kw)
    print("[loop] done", flush=True)


if __name__ == "__main__":
    main()
