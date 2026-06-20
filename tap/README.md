# TAP v3 — critique-corrected causal-update-utility prediction

**Goal:** forecast how much one **GRPO update** will improve held-out MATH
performance (and not cause generic drift), so we can *select* good updates / screen
cohorts before paying for a full RL run. Policy: `Qwen2.5-Math-1.5B-Instruct`
(weak at MATH-500 → headroom). MATH-500 stays sealed for the final demo.

This is the synthesis of v1 (label-efficient separate predictor), v2 (reset-per-
cohort battery; diversity/redundancy features), the TAP spec (multi-anchor chains,
policy fingerprint, KL-drift penalty, selection metrics), and — crucially — the
**TAP critiques**.

## How v3 answers each TAP critique

| TAP critique | v3 fix |
|---|---|
| 300k-param net on ~100 labels overfits | **GBDT / ridge** (`predictor.py`), monotone-constrained; no attention net |
| Underpowered eval (3 folds, n=1 demo) | **leave-one-chain-out + LOOCV + bootstrap CIs**; selection-lift reported with CIs |
| Hand-weighted NLL composite; NLL≠accuracy | **accuracy is the default label**; NLL is a logged dense proxy; `utility` weights are explicit (`labels.UtilityWeights`) and validated, never opaque |
| Per-candidate "matched probe" breaks ranking | one **COMMON global probe** for every candidate at an anchor |
| Single-step utility ≈ noise; "nonzero variance" too weak | **`gate.py`** measures within- vs between-cohort variance (ICC/SNR) from multi-seed runs; **M GRPO steps** per branch |
| KL penalty negligible at 1 step | one-sided KL drift over **M** steps, vs frozen base (adapters disabled — no 2nd model) |
| Gradient features undercut "predict before training" | **cheap rollout features are primary**; gradient-alignment is an optional **baseline** only; a **cost-quality frontier** (a-priori → rollout → +gradient) is measured, not assumed |
| Greedy top-1 ignores diversity | report **top-k selection lift** (not just top-1); quality cohorts include redundancy/duplication |
| Zero-variance collapse not isolated | `frac_nondegenerate` is a first-class feature; cohorts span the regime |
| Leakage | strictly **pre-update** features; labels stored apart; feature whitelist in `build_xy` |
| Model too strong (8B) → tiny lift | weaker **1.5B** math model with real MATH-500 headroom |

## Package

- `cohorts.py` — random / difficulty / variance-decoupled / subject / **label-noise**
  / **duplication** cohorts (pure stdlib).
- `features.py` — pre-update rollout features: learnability (reward variance,
  `frac_nondegenerate`), familiarity (mean log-prob, entropy, quantiles, confidence
  slope), redundancy + **target-similarity to the probe** (pure stdlib).
- `labels.py` — `LiftLabel` (acc / nll / one-sided KL drift) + explicit `UtilityWeights`.
- `metrics.py` — regression **and** selection metrics (within-anchor Spearman,
  pairwise accuracy, top-1 regret, top-k selection lift, bootstrap CIs).
- `predictor.py` — monotone GBDT/ridge; leave-one-chain-out + LOOCV; baselines
  (difficulty / reward / redundancy / gradient-alignment / best-single); SHAP.
- `gate.py` — signal-vs-noise ICC/SNR analysis (run this **first**).
- `battery.py` — the GRPO battery (multi-anchor chains, random-advance, common
  probe + KL drift, fingerprint, noise seeds, label-noise rewards). Lazy torch.
- `data_probes.py` — MATH splits + global/fingerprint/generic probes (reuses
  `math_loop`).
- `../run_tap_pod.py` — single-GPU Prime Intellect launcher.
- `../scripts/synth_labels.py` — synthetic labels to exercise predictor/gate off-GPU.

## Run order

```bash
# 0. laptop sanity (no GPU)
python -m unittest tests.test_tap -v
python scripts/synth_labels.py --out outputs/synth/labels.jsonl --noise 0.01
python -m tap.gate      --labels outputs/synth/labels.jsonl
python -m tap.predictor --labels outputs/synth/labels.jsonl --scheme logo

# 1. THE GATE (cheap pod run): does single-anchor accuracy lift beat noise?
python run_tap_pod.py --gpu-count 1 --max-cohorts 4 --seeds 4 \
    --probe-size 24 --grpo-steps 4 --output-dir outputs/tap_noise
python -m tap.gate --labels outputs/tap_noise/labels.jsonl
#   verdict must be usable/strong before scaling. If "mostly_noise":
#   raise --grpo-steps / --probe-k / --cohort-size.

# 2. full battery with chains (enables leave-one-chain-out)
python run_tap_pod.py --gpu-count 1 --n-chains 3 --anchors-per-chain 4 \
    --output-dir outputs/tap_full
python -m tap.predictor --labels outputs/tap_full/labels.jsonl --scheme logo --explain

# 3. (later) second model -> transfer; end-to-end 4-step selection vs reward/random.
```

## Success criterion

Selection-focused, not regression error: TAP's top-ranked candidates have higher
true held-out accuracy lift than random and reward-only selection on **held-out
chains** (with bootstrap CIs), and the learned model beats difficulty-only and the
best single feature. The honest gate (`gate.py`) must pass first.

## Measurement & limitations (honest notes)

**Lift is a *batch* quantity, not a per-trace one.** A single GRPO trace has no
group-relative advantage and a true lift below any eval's noise floor, so it is
*unmeasurable in principle* — not a predictor-precision problem. The predictor's unit
is therefore a **cohort / candidate update batch** (also the real use case: scoring a
purchased data shard). `predictor.predict_interval` returns a **split-conformal**
interval and flags `below_resolution` (interval straddles 0) so the model *honestly
says "can't call the sign"* instead of emitting a fake number; `evaluate(...)["conformal"]`
reports `coverage` and `indeterminate_sign_rate`. Ways to *legitimately* tighten this
are all batch-level: aggregate features (√N feature-noise reduction), the ranking
objective (needs only relative order), monotone priors, and more seeds — none make a
single trace significant.

**LR amplification is a deliberate hackathon compromise.** To surface a lift signal
above noise *cheaply* (small cohorts, 1 GPU) we run a **boosted learning rate**
(`--lr 5e-4`, ~50× a production fine-tune). Lift scales ~linearly with LR in the
small-update regime, so this *amplifies* the label while **preserving the relative
ordering of cohorts** — which is exactly what the predictor (within-anchor ranking)
and gate (ICC) consume. The IS-clip + grad-clip + KL-to-reference terms keep it below
the instability ceiling, and the **label-noise dose-response** cohorts separate by the
*direction* of the gradient (clean→correct reward, poisoned→wrong reward), so the
good-vs-bad-data signal is robust to amplification.

*In a production setting* you would instead use a realistic LR (≈1–5e-5) with **larger
cohorts and more seeds** to make the (smaller) lift statistically significant, and
rescale predictions accordingly. Absolute lift magnitudes reported here are inflated;
the **rankings and the feature→lift directions are the transferable result.**
