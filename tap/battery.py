"""In-process GRPO battery: per-candidate causal-update-utility labels.

For each policy anchor along a short main chain, branch many candidate cohorts
from the **identical** LoRA snapshot, apply M GRPO steps, and measure the update's
held-out utility on a **common probe**:

    acc/NLL on a fixed global MATH probe  +  generic-KL drift penalty.

Then advance the main chain by applying one *randomly chosen* candidate (so the
state history isn't biased toward a heuristic). Repeat across N independent chains
so the predictor can be evaluated leave-one-chain-out.

Design choices (from the TAP critiques):
* COMMON probe for every candidate at an anchor -> within-anchor ranking is fair.
* Accuracy is the real label; NLL is a dense proxy; both are logged. KL drift is a
  one-sided penalty measured against the frozen base (adapters disabled) -- no 2nd
  model load.
* Features are computed from a NO-GRADIENT rollout (pre-update) and stored apart
  from the after-update labels (leakage guard).
* ``--seeds > 1`` repeats each candidate for the signal-vs-noise gate.

Heavy deps (torch/transformers/peft) are imported lazily; ``group_advantages`` and
the record assembly are pure and unit-test on a laptop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

from tap import features as F
from tap.cohorts import Cohort, build_all_cohorts, read_cohorts
from tap.labels import LiftLabel, UtilityWeights


def group_advantages(rewards: Sequence[float], eps: float = 1e-8) -> list[float]:
    """GRPO group-relative advantages; zeros when the group has no reward variance."""

    if not rewards:
        return []
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    if std <= eps:
        return [0.0 for _ in rewards]
    return [(r - mean) / (std + eps) for r in rewards]


def corrupt_answer(answer: str) -> str:
    """Deterministic wrong answer for label-noise cohorts (poisons the reward)."""

    a = str(answer).strip()
    return (a + "1") if a and a[-1].isdigit() else (a + "0") if a else "0"


@dataclass
class BatteryConfig:
    model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    domain_name: str = "math"     # math | code | science (selects data + verifier + render)
    device: str = "cuda"
    dtype: str = "bfloat16"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lr: float = 1e-5
    grpo_steps: int = 4          # M GRPO steps per candidate branch
    group_size: int = 8          # completions per prompt
    max_new_tokens: int = 512
    temperature: float = 1.0
    max_seq_len: int = 4096
    probe_k: int = 4             # samples per probe item (sampled eval only)
    eval_greedy: bool = True     # deterministic probe eval -> kills acc sampling noise
    acc_eval: bool = True        # measure greedy accuracy (slow); off => NLL-only gate
    grad_clip: float = 1.0
    micro_batch: int = 8         # padded seqs per fwd/bwd pass (memory <-> throughput)
    kl_beta: float = 0.04        # KL-to-reference weight (GRPO objective)
    clip_eps: float = 0.2        # importance-ratio clip (PPO/GRPO surrogate)
    adv_eps: float = 1e-8
    # chain / anchor structure
    n_chains: int = 1
    anchors_per_chain: int = 1
    seeds: int = 1               # repeats per candidate (noise gate when > 1)
    seed: int = 0
    w: UtilityWeights = None  # type: ignore

    def __post_init__(self):
        if self.w is None:
            self.w = UtilityWeights()


_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ---- model + state plumbing (lazy torch) --------------------------------------


def _load_policy(cfg: BatteryConfig):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.dtype]
    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dt, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                                             lora_dropout=cfg.lora_dropout, target_modules=_LORA_TARGETS,
                                             task_type="CAUSAL_LM"))
    model.to(cfg.device)
    return model, tok


def _snapshot(model) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _reset(model, snap: dict) -> None:
    import torch

    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in snap:
                p.copy_(snap[n])


def _question(row: dict) -> str:
    return row.get("question") or row["problem"]


# ---- generation + scoring -----------------------------------------------------


def _render(tok, q: str, cfg: "BatteryConfig") -> str:
    from tap.domains import chat_render, get_domain

    return chat_render(tok, q, get_domain(cfg.domain_name).system)


def _sample(model, tok, q: str, cfg: BatteryConfig, n: int, *, greedy: bool = False):
    import torch

    enc = tok(_render(tok, q, cfg), return_tensors="pt").to(cfg.device)
    gkw = dict(max_new_tokens=cfg.max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    if greedy:  # deterministic: one completion, no sampling variance in the eval
        gkw.update(do_sample=False, num_return_sequences=1)
    else:
        gkw.update(do_sample=True, temperature=cfg.temperature, num_return_sequences=n)
    with torch.no_grad():
        gen = model.generate(**enc, **gkw)
    plen = enc.input_ids.shape[1]
    eos = tok.eos_token_id
    comps = []
    for i in range(gen.shape[0]):
        ids = gen[i, plen:].tolist()
        if eos is not None and eos in ids:
            ids = ids[: ids.index(eos) + 1]
        comps.append(ids)
    return enc.input_ids[0].tolist(), comps


def _reward(text: str, item: dict, cfg: "BatteryConfig") -> float:
    from tap.domains import get_domain

    return get_domain(cfg.domain_name).reward(text, item)


def _completion_signals(model, tok, prompt_ids: list[int], comp_ids: list[int], cfg: BatteryConfig) -> dict:
    """No-grad per-completion familiarity signals (mean logprob/entropy, slope)."""

    import torch
    import torch.nn.functional as Fnn

    if not comp_ids:
        return {}
    full = torch.tensor([prompt_ids + comp_ids], device=cfg.device)[:, : cfg.max_seq_len]
    plen = len(prompt_ids)
    with torch.no_grad():
        logits = model(input_ids=full).logits[0, :-1, :].float()
    logp = Fnn.log_softmax(logits, dim=-1)
    tgt = full[0, 1:]
    c0 = max(plen - 1, 0)
    comp_logp = logp[c0:].gather(-1, tgt[c0:].unsqueeze(-1)).squeeze(-1)
    ent = -(logp[c0:].exp() * logp[c0:]).sum(-1)
    n = comp_logp.numel()
    if n == 0:
        return {}
    half = max(n // 2, 1)
    return {
        "mean_logprob": float(comp_logp.mean()),
        "mean_entropy": float(ent.mean()),
        "early_logprob": float(comp_logp[:half].mean()),
        "late_logprob": float(comp_logp[half:].mean()) if n > half else float(comp_logp[-1]),
        "completion_tokens": int(n),
    }


def eval_accuracy(model, tok, rows: Sequence[dict], cfg: BatteryConfig) -> float:
    """Held-out pass-rate. Greedy (deterministic) eval removes sampling noise so
    that lift = real model change, not probe-sampling variance (v1's #1 noise)."""

    model.eval()
    k = 1 if cfg.eval_greedy else cfg.probe_k
    correct = total = 0
    for row in rows:
        _, comps = _sample(model, tok, _question(row), cfg, k, greedy=cfg.eval_greedy)
        for c in comps:
            correct += int(_reward(tok.decode(c, skip_special_tokens=True), row, cfg) > 0.5)
            total += 1
    return correct / max(total, 1)


def eval_nll(model, tok, rows: Sequence[dict], cfg: BatteryConfig) -> float:
    """Teacher-forced NLL (nats/token) on gold solutions -- micro-batched (dense,
    cheap; the gate's primary signal). NLL = -mean(token logp) over gold tokens."""

    model.eval()
    fulls, plens = [], []
    for row in rows:
        prompt = _render(tok, _question(row), cfg)
        target = (row.get("solution") or f"\\boxed{{{row['answer']}}}").strip()
        pids = tok(prompt, add_special_tokens=False).input_ids
        ids = tok(prompt + target + (tok.eos_token or ""), add_special_tokens=False).input_ids[: cfg.max_seq_len]
        if len(ids) <= len(pids):
            continue
        fulls.append(ids)
        plens.append(len(pids))
    if not fulls:
        return 0.0
    lps = _token_logps_batch(model, fulls, plens, tok.pad_token_id, cfg, grad=False)
    tot = sum(-float(lp.sum()) for lp in lps)
    n = sum(int(lp.numel()) for lp in lps)
    return tot / max(n, 1)


def eval_generic_kl(model, tok, rows: Sequence[dict], cfg: BatteryConfig) -> float:
    """Mean per-token KL(current || frozen-base) on generic prompts (drift).

    The frozen base is the same model with LoRA adapters disabled -- no 2nd load.
    """

    import torch
    import torch.nn.functional as Fnn

    model.eval()
    tot_kl = tot_tok = 0
    for row in rows:
        text = row.get("text") or (_question(row) + " " + (row.get("solution") or ""))
        ids = tok(text, add_special_tokens=False).input_ids[: cfg.max_seq_len]
        if len(ids) < 2:
            continue
        it = torch.tensor([ids], device=cfg.device)
        with torch.no_grad():
            cur = Fnn.log_softmax(model(input_ids=it).logits[0, :-1, :].float(), dim=-1)
            with model.disable_adapter():
                ref = Fnn.log_softmax(model(input_ids=it).logits[0, :-1, :].float(), dim=-1)
        kl = (cur.exp() * (cur - ref)).sum(-1)  # KL(cur||ref) per position
        tot_kl += float(kl.sum())
        tot_tok += kl.numel()
    return tot_kl / max(tot_tok, 1)


def fingerprint(model, tok, rows: Sequence[dict], cfg: BatteryConfig) -> dict:
    """Compact policy-competence summary: mean gold NLL + mean entropy on a fixed set."""

    import torch
    import torch.nn.functional as Fnn

    model.eval()
    nlls, ents = [], []
    for row in rows:
        prompt = _render(tok, _question(row), cfg)
        pids = tok(prompt, add_special_tokens=False).input_ids
        comp = tok((row.get("solution") or f"\\boxed{{{row['answer']}}}"), add_special_tokens=False).input_ids
        sig = _completion_signals(model, tok, pids, comp, cfg)
        if sig:
            nlls.append(-sig["mean_logprob"])
            ents.append(sig["mean_entropy"])
    return {
        "fingerprint_nll": statistics.fmean(nlls) if nlls else float("nan"),
        "fingerprint_entropy": statistics.fmean(ents) if ents else float("nan"),
    }


# ---- fixed-rollout, batched GRPO (review P0: fidelity + speed) -----------------


@dataclass
class Traj:
    """One sampled completion + its fixed-rollout bookkeeping (generated once)."""

    prompt_ids: list
    comp_ids: list
    reward: float
    group_id: str
    advantage: float = 0.0
    old_token_logp: tuple = ()   # per-token logp at collection time (IS anchor)
    n_tokens: int = 0
    mean_logprob: float = 0.0
    early_logprob: float = 0.0
    late_logprob: float = 0.0


def _token_logps_batch(model, fulls, prompt_lens, pad_id, cfg: BatteryConfig, *, grad: bool):
    """Per-completion-token logp for each (prompt+comp) sequence, micro-batched.

    Right-pad + causal => real-token logits are unaffected by padding. Uses
    ``gather - logsumexp`` (no full ``log_softmax``) to bound memory. Returns a
    list of 1-D tensors (one per sequence), length = #completion tokens.
    """

    import torch

    ctx = torch.enable_grad() if grad else torch.no_grad()
    out = []
    with ctx:
        for i in range(0, len(fulls), cfg.micro_batch):
            chunk = fulls[i : i + cfg.micro_batch]
            plens = prompt_lens[i : i + cfg.micro_batch]
            width = max(len(s) for s in chunk)
            ids = torch.full((len(chunk), width), pad_id, dtype=torch.long, device=cfg.device)
            attn = torch.zeros((len(chunk), width), dtype=torch.long, device=cfg.device)
            for j, s in enumerate(chunk):
                ids[j, : len(s)] = torch.tensor(s, dtype=torch.long, device=cfg.device)
                attn[j, : len(s)] = 1
            logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1, :].float()
            tgt = ids[:, 1:]
            tok_lp = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
            for j, s in enumerate(chunk):
                out.append(tok_lp[j, max(plens[j] - 1, 0) : len(s) - 1])
    return out


def collect_trajectories(model, tok, cohort_rows, cfg: BatteryConfig, noisy_ids: set, probe_unigrams: dict):
    """Generate each prompt's group ONCE; compute features + IS anchors from the
    SAME trajectories the update trains on (fixed-rollout consistency)."""

    model.eval()
    trajs: list[Traj] = []
    tok_lists: list[list[int]] = []
    for row in cohort_rows:
        pids, comps = _sample(model, tok, _question(row), cfg, cfg.group_size)
        tok_lists.append(tok(_question(row), add_special_tokens=False).input_ids)
        item = {**row, "answer": corrupt_answer(row.get("answer", ""))} if str(row["id"]) in noisy_ids else row
        for c in comps:
            if not c:
                continue
            trajs.append(Traj(prompt_ids=list(pids), comp_ids=list(c),
                              reward=_reward(tok.decode(c, skip_special_tokens=True), item, cfg),
                              group_id=str(row["id"])))
    for ts in {t.group_id: None for t in trajs}:  # group-relative advantages
        members = [t for t in trajs if t.group_id == ts]
        for t, a in zip(members, group_advantages([m.reward for m in members], cfg.adv_eps)):
            t.advantage = a
    rollouts: list[dict] = []
    if trajs:
        fulls = [t.prompt_ids + t.comp_ids for t in trajs]
        plens = [len(t.prompt_ids) for t in trajs]
        for t, lp in zip(trajs, _token_logps_batch(model, fulls, plens, tok.pad_token_id, cfg, grad=False)):
            vals = lp.detach().float().cpu().tolist()
            t.old_token_logp = tuple(vals)
            t.n_tokens = len(vals)
            if vals:
                half = max(len(vals) // 2, 1)
                t.mean_logprob = sum(vals) / len(vals)
                t.early_logprob = sum(vals[:half]) / half
                t.late_logprob = sum(vals[half:]) / max(len(vals) - half, 1)
            rollouts.append({"group_id": t.group_id, "reward": t.reward, "completion_tokens": t.n_tokens,
                             "mean_logprob": t.mean_logprob, "early_logprob": t.early_logprob,
                             "late_logprob": t.late_logprob})
    stats = F.summarize_rollouts(rollouts).as_dict()
    tsim = F.target_similarity(tok_lists, probe_unigrams)
    return trajs, stats, tsim, len(trajs)


def grpo_train(model, trajs, cfg: BatteryConfig, optimizer, pad_id) -> dict:
    """``cfg.grpo_steps`` epochs over the FIXED trajectories: batched clipped-IS
    surrogate + KL-to-frozen-reference (adapters disabled), micro-batched grad
    accumulation. The reference is computed once (the rollouts don't change)."""

    import torch

    active = [t for t in trajs if t.advantage != 0.0 and t.n_tokens > 0]
    if not active:
        return {"mean_reward": statistics.fmean([t.reward for t in trajs]) if trajs else 0.0, "kl": 0.0, "n_contrib": 0}
    fulls = [t.prompt_ids + t.comp_ids for t in active]
    plens = [len(t.prompt_ids) for t in active]
    with torch.no_grad(), model.disable_adapter():
        ref = [r.detach() for r in _token_logps_batch(model, fulls, plens, pad_id, cfg, grad=False)]
    n = len(active)
    last_kl = 0.0
    for _epoch in range(cfg.grpo_steps):
        model.train()
        optimizer.zero_grad()
        total_kl = 0.0
        contrib = 0
        for i in range(0, n, cfg.micro_batch):
            idx = list(range(i, min(i + cfg.micro_batch, n)))
            cur = _token_logps_batch(model, [fulls[j] for j in idx], [plens[j] for j in idx], pad_id, cfg, grad=True)
            loss = None
            for k, j in enumerate(idx):
                t = active[j]
                c_lp, r_lp = cur[k], ref[j]
                o_lp = torch.tensor(t.old_token_logp, device=cfg.device)
                m = min(c_lp.numel(), o_lp.numel(), r_lp.numel())
                if m == 0:
                    continue
                c_lp, o_lp, r_lp = c_lp[:m], o_lp[:m], r_lp[:m]
                ratio = torch.exp(c_lp - o_lp)
                pg = -torch.min(ratio * t.advantage,
                                torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * t.advantage)
                kl = torch.exp(r_lp - c_lp) - (r_lp - c_lp) - 1.0   # k3 KL(policy||ref) >= 0
                term = (pg + cfg.kl_beta * kl).mean()
                loss = term if loss is None else loss + term
                total_kl += float(kl.mean())
                contrib += 1
            if loss is not None:
                (loss / n).backward()
        if contrib > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
            optimizer.step()
        last_kl = total_kl / max(contrib, 1)
    return {"mean_reward": statistics.fmean([t.reward for t in trajs]), "kl": last_kl, "n_contrib": len(active)}


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as h:
        h.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=float) + "\n")


def completed_keys(path: Path) -> set:
    """(chain, anchor, candidate, seed) keys already in labels.jsonl (resume support)."""

    keys: set = set()
    p = Path(path)
    if p.exists():
        for line in p.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((r.get("chain_id"), r.get("anchor_index"), r.get("candidate_id"), r.get("seed")))
    return keys


def _write_status(path: Path, **fields) -> None:
    """Heartbeat: a tiny status.json next to the labels so progress is observable."""

    try:
        (Path(path).parent / "status.json").write_text(
            json.dumps({"ts": time.time(), **fields}, default=float), encoding="utf-8"
        )
    except OSError:
        pass


# ---- the battery loop ---------------------------------------------------------


def run_battery(cfg: BatteryConfig, cohorts: Sequence[Cohort], probes: dict, rows_by_id: dict, out_path: Path) -> list[dict]:
    import torch

    model, tok = _load_policy(cfg)
    global_probe = probes["global"]
    generic_probe = probes.get("generic", [])
    fp_rows = probes.get("fingerprint", [])
    probe_unigrams = F._unigrams(tok(_question(r), add_special_tokens=False).input_ids for r in global_probe)

    base_snapshot = _snapshot(model)
    records: list[dict] = []
    done = completed_keys(out_path)
    if done:
        print(f"[tap] resuming: {len(done)} labels already present; skipping those", flush=True)
    n_anchor_steps = max(cfg.anchors_per_chain - 1, 1)
    print(f"[tap] model={cfg.model_name} chains={cfg.n_chains} anchors={cfg.anchors_per_chain} "
          f"cohorts={len(cohorts)} seeds={cfg.seeds}", flush=True)

    for chain in range(cfg.n_chains):
        _reset(model, base_snapshot)
        for anchor in range(cfg.anchors_per_chain):
            anchor_snap = _snapshot(model)
            fp = fingerprint(model, tok, fp_rows, cfg) if fp_rows else {"fingerprint_nll": float("nan"), "fingerprint_entropy": float("nan")}
            acc_before = eval_accuracy(model, tok, global_probe, cfg) if cfg.acc_eval else 0.0
            nll_before = eval_nll(model, tok, global_probe, cfg)
            kl_before = eval_generic_kl(model, tok, generic_probe, cfg) if generic_probe else 0.0
            print(f"[tap] anchor c{chain} a{anchor} ready | nll_before={nll_before:.4f} "
                  f"acc_before={acc_before:.3f} cohorts={len(cohorts)}x{cfg.seeds}seeds", flush=True)

            for cohort in cohorts:
                cohort_rows = [rows_by_id[p] for p in cohort.prompt_ids if p in rows_by_id]
                if not cohort_rows:
                    continue
                noisy = set(map(str, cohort.meta.get("noisy_ids", [])))
                for seed in range(cfg.seeds):
                    if (chain, anchor, cohort.name, seed) in done:
                        continue  # resume: already collected
                    torch.manual_seed(cfg.seed + 1000 * chain + 7 * anchor + seed)
                    _reset(model, anchor_snap)
                    t0 = time.time()
                    print(f"[tap] c{chain} a{anchor} {cohort.name:22s} s{seed} collecting+training...", flush=True)
                    trajs, feats, tsim, n_feat = collect_trajectories(model, tok, cohort_rows, cfg, noisy, probe_unigrams)
                    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
                    train_stats = grpo_train(model, trajs, cfg, opt, tok.pad_token_id)
                    acc_after = eval_accuracy(model, tok, global_probe, cfg) if cfg.acc_eval else 0.0
                    nll_after = eval_nll(model, tok, global_probe, cfg)
                    kl_after = eval_generic_kl(model, tok, generic_probe, cfg) if generic_probe else 0.0
                    label = LiftLabel(acc_before, acc_after, nll_before, nll_after, kl_before, kl_after)
                    rec = {
                        "schema_version": 3,
                        "domain": cfg.domain_name,
                        "model_name": cfg.model_name,
                        "chain_id": chain,
                        "anchor_index": anchor,
                        "candidate_id": cohort.name,
                        "seed": seed,
                        "cohort": cohort.to_json(),
                        "reward_summary": feats,
                        "target_similarity": tsim,
                        "step_frac": anchor / n_anchor_steps,
                        **fp,
                        **label.as_dict(),
                        "utility": label.utility(cfg.w),
                        "mean_reward": train_stats["mean_reward"],
                        "kl_train": train_stats["kl"],
                        "n_contrib": train_stats["n_contrib"],
                        "rollout_count": n_feat + 2 * len(global_probe),
                        "wall_clock_s": time.time() - t0,
                    }
                    _append(out_path, rec)
                    records.append(rec)
                    _write_status(out_path, chain=chain, anchor=anchor, candidate=cohort.name, seed=seed,
                                  labels_done=len(done) + len(records), last_lift_acc=rec["lift_acc"],
                                  last_lift_nll=rec["lift_nll"])
                    print(f"[tap] c{chain} a{anchor} {cohort.name:22s} s{seed} "
                          f"lift_acc={label.lift_acc:+.4f} lift_nll={label.lift_nll:+.4f} "
                          f"acc {acc_before:.3f}->{acc_after:.3f} t={rec['wall_clock_s']:.0f}s", flush=True)

            # advance the main chain by a RANDOM candidate (unbiased history)
            _reset(model, anchor_snap)
            if anchor < cfg.anchors_per_chain - 1:
                import random

                pick = random.Random(cfg.seed + chain * 31 + anchor).choice(list(cohorts))
                pick_rows = [rows_by_id[p] for p in pick.prompt_ids if p in rows_by_id]
                noisy = set(map(str, pick.meta.get("noisy_ids", [])))
                trajs, _, _, _ = collect_trajectories(model, tok, pick_rows, cfg, noisy, probe_unigrams)
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
                grpo_train(model, trajs, cfg, opt, tok.pad_token_id)
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/tap"))
    p.add_argument("--cohorts", type=Path, default=None)
    p.add_argument("--pass-rates", type=Path, default=None)
    p.add_argument("--output", type=Path, default=Path("outputs/tap/labels.jsonl"))
    p.add_argument("--model-name", default=BatteryConfig.model_name)
    p.add_argument("--domain", default="math", choices=("math", "code", "science"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--grpo-steps", type=int, default=4)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="sampling temperature for rollouts; >1 raises within-group reward variance (GRPO signal)")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--probe-size", type=int, default=64)
    p.add_argument("--probe-k", type=int, default=4)
    p.add_argument("--eval-sample", dest="eval_greedy", action="store_false",
                   help="sample probe eval (noisier) instead of deterministic greedy")
    p.set_defaults(eval_greedy=True)
    p.add_argument("--no-acc-eval", dest="acc_eval", action="store_false",
                   help="skip greedy accuracy eval (NLL-only gate; much faster)")
    p.set_defaults(acc_eval=True)
    p.add_argument("--cohort-size", type=int, default=8)
    p.add_argument("--n-random", type=int, default=8)
    p.add_argument("--n-chains", type=int, default=1)
    p.add_argument("--anchors-per-chain", type=int, default=1)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--max-cohorts", type=int, default=0)
    p.add_argument("--shard", default=None, help="parallel split 'i/n' (worker i of n takes cohorts[i::n])")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from tap.domains import get_domain

    args = parse_args(argv)
    bundle = get_domain(args.domain).load_splits(args.data_dir, probe_size=args.probe_size,
                                                 fingerprint_size=16, seed=args.seed)
    rows_by_id = {r["id"]: r for r in bundle["train_rows"]}

    if args.cohorts is not None:
        cohorts = read_cohorts(args.cohorts)
    else:
        pass_rates = None
        if args.pass_rates is not None:
            pass_rates = {str(k): float(v) for k, v in json.loads(Path(args.pass_rates).read_text()).items()}
        cohorts = build_all_cohorts(bundle["train_rows"], size=args.cohort_size, pass_rates=pass_rates,
                                    n_random=args.n_random, seed=args.seed)
    if args.max_cohorts:
        cohorts = cohorts[: args.max_cohorts]
    if args.shard:  # parallel workers: each takes cohorts[i::n]
        i, n = (int(x) for x in args.shard.split("/"))
        cohorts = cohorts[i::n]

    cfg = BatteryConfig(model_name=args.model_name, domain_name=args.domain, device=args.device, dtype=args.dtype,
                        grpo_steps=args.grpo_steps, group_size=args.group_size, max_new_tokens=args.max_new_tokens,
                        probe_k=args.probe_k, eval_greedy=args.eval_greedy, acc_eval=args.acc_eval, n_chains=args.n_chains,
                        anchors_per_chain=args.anchors_per_chain, seeds=args.seeds, lr=args.lr, seed=args.seed,
                        temperature=args.temperature)
    run_battery(cfg, cohorts, bundle["probes"], rows_by_id, args.output)


if __name__ == "__main__":
    main()
