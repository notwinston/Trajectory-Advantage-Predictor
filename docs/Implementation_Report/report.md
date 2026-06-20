# TAP v1 — Trajectory Advantage Predictor: Implementation Report

**Wave 2 (integration + Prime Intellect hardening + handoff).** PDF engine:
**reportlab 5.0.0** (`build_report_pdf.py`; no LaTeX toolchain is installed on the
ARM64 build host, so the `.tex` source ships alongside a reportlab-rendered PDF,
matching the repo's existing `tap.report` convention).

## 1. Goal

TAP predicts a single scalar, `predicted_utility_points`, for a candidate GRPO
update batch applied to a Qwen3-8B LoRA policy. The scalar estimates how much
applying that batch will improve held-out MATH performance while lightly
penalizing unrelated policy drift. Positive = expected to help, ~0 = little
effect, negative = expected to hurt/drift. No good/neutral/harmful label is used.

The hypothesis under test: *a candidate update's usefulness is predictable from
its reward, familiarity (probability), gradient direction, current policy state,
and similarity to recently reinforced updates.* Success = TAP selects candidates
with higher average true utility than random and probability-only on held-out
states; the strong result is also beating reward-only and the no-history model.

## 2. Architecture

Three cooperating layers, separated so everything except GPU collection runs and
is tested on CPU:

1. **Data collection (`math_loop/`)** — runs on a 4–5×H100 Prime Intellect pod.
   `tap_controller` walks 2 chains × 6 states × 6 candidates. At each state it
   saves the before-state (LoRA adapter + optimizer + step/seed/lr/grpo_beta),
   measures the probes, records the 16-value policy fingerprint, generates 6
   candidate GRPO batches (2 prompts × 4 completions = 8 trajectories each),
   branches every candidate from the byte-identical before-state on a worker GPU,
   labels it, and advances the main chain with a **seeded-random** candidate (so
   the collected history is unbiased). `branch.py` runs one GRPO step via prime-rl
   resume and writes raw artifacts (`before/`, `cand_<k>/`, `state.json`);
   `features.py` converts the raw tree into the four Parquet tables;
   `tap_probes.py` computes teacher-forced probe NLL and generic KL drift.

2. **Launcher + safety (`run_prime_rl_math_loop.py`, `reap_pods.py`)** — Wave 2.
   See §8. Provisions the pod, pins prime-rl, runs collection + featurization,
   downloads artifacts, validates them, and guarantees teardown.

3. **TAP model + evaluation (`tap/`)** — CPU. `dataset.py` loads the Parquet,
   builds chain-wise splits and ablation masks; `model.py` is SmallTAP;
   `baselines.py` holds the heuristic selectors and learned baselines;
   `train.py` fits SmallTAP; `eval.py` computes within-state ranking metrics;
   `run_all.py` is the end-to-end entry point that emits `results.csv` +
   `report.{md,tex,pdf}`.

## 3. Data schema (four Parquet tables)

A frozen schema contract is enforced by `tap/schema.py --validate <dir>`:

- **states.parquet** — one row per main-chain state: ids, step, seed, checkpoint
  / optimizer hashes, learning rate, grpo_beta, clip_range, lora_rank, the three
  *before* probe measurements, Adam moment norms, the 16-value `policy_fingerprint`
  (NLL + entropy on 8 fixed held-out MATH prompts), and `history_candidate_ids`.
- **trajectories.parquet** — one row per completion: reward components, advantage,
  sequence length, log-probability statistics and quantiles, entropy statistics,
  early/late confidence slope, old→current and current→reference log-ratios,
  clipped-token fraction, and the trajectory embedding.
- **candidates.parquet** — the main TAP training table, one row per candidate
  branch: reward/advantage mean+std, probability statistics, mean entropy, mean
  sequence length, 256-d candidate embedding, 64-d gradient sketch, gradient and
  update norms, semantic/gradient similarity to history, the three *after* probe
  measurements, `matched_gain` / `global_gain` / `incremental_generic_kl`, the
  `utility_points` label, exact-match before/after diagnostics, and
  `is_selected_for_main_chain`.
- **history.parquet** — one row per historical update attached to a state (last 4
  applied): relative age, candidate embedding, gradient sketch, reward/advantage
  means, log-probability/entropy means, update norm, training-loss change, and
  candidate log-probability change. The historical branch's own utility label is
  **not** an input.

### Utility label

```
matched_gain          = matched_probe_nll_before - matched_probe_nll_after
global_gain           = global_probe_nll_before  - global_probe_nll_after
incremental_generic_kl= generic_kl_after - generic_kl_before
utility_points        = 1000 * (0.8*matched_gain + 0.2*global_gain
                                - 0.03*max(incremental_generic_kl, 0))
```

All NLL/KL values are averages in nats per non-padding token. Exact-match is kept
for diagnostics only (too sparse after one small GRPO step).

## 4. Features TAP uses

- **State:** step, learning rate, grpo_beta, Adam moment norms, policy fingerprint.
- **Candidate:** reward/advantage mean+std, mean/geometric/arithmetic probability,
  entropy stats, log-prob quantiles, early-late confidence slope, sequence length,
  policy/reference log-ratio, 256-d candidate embedding, 64-d gradient sketch,
  gradient norm, estimated update norm.
- **History:** last-4 embeddings + gradient sketches, relative ages, reward/
  probability stats, semantic similarity, gradient similarity.

Interpretation: probability = familiarity, reward/advantage = desirability,
gradient = expected parameter movement, history similarity = already reinforced.

## 5. Models

**SmallTAP** (`tap/model.py`) — numeric-feature MLP + shallow history attention:
candidate-embedding projection (256→64), gradient-sketch projection (64→32),
numeric MLP (16→64), state MLP (26→32), history records projected to 64, one
4-head cross-attention layer from candidate to the 4 history slots, and a
two-layer MLP head (hidden 128) emitting one scalar. **Trainable parameters:
109,537** — under the 250k target.

**Baselines** (all evaluated for the same within-state ranking):

- *Heuristic selectors:* random, highest reward mean, highest advantage mean,
  highest geometric / arithmetic mean probability, reward × surprisal, semantic
  novelty, gradient norm, gradient alignment.
- *Learned baselines:* ridge regression, gradient-boosted trees, no-history MLP,
  numeric-only, candidate-only.

Per the spec, if the attention model does not beat the simpler learned baselines
on the real collection, the simpler model is reported as TAP v1.

## 6. Training & data split

Loss = Huber on standardized `utility_points` + 0.5 × within-state pairwise
ranking loss + small weight decay (scalar objective only; no classification, no
error×probability weighting). Split = train on chain 0 / test on chain 1, then
swap, report the average of both directions; all candidates from one state stay
in the same split. With ≥128 labels, the last two states of each training chain
are held out for early stopping.

## 7. Evaluation metrics & current (synthetic placeholder) numbers

Primary metrics: within-state Spearman, pairwise ranking accuracy, top-1 regret,
mean true utility of selected candidates, and lift over random / reward-only /
probability-only.

The collection is **blocked on the Prime Intellect API key** (see §8), so the
GPU labels do not exist yet. The numbers below are the **placeholder** produced
by `tap.run_all` on the synthetic 72-label dataset (`outputs/tap_synth_72`) — a
plumbing check, **not** a scientific result:

| Model | Spearman | Pair acc | Mean true utility | Lift vs random | Lift vs reward | Lift vs prob |
|---|---|---|---|---|---|---|
| **TAP (SmallTAP)** | 0.738 | 0.817 | 34.42 | +39.35 | +12.57 | +48.30 |
| ridge | 0.829 | 0.861 | 35.79 | +40.72 | +13.94 | +49.67 |
| candidate-only | 0.843 | 0.872 | 36.59 | +41.52 | +14.74 | +50.46 |
| no-history MLP | 0.481 | 0.683 | 32.92 | +37.85 | +11.07 | +46.80 |
| reward-only | 0.233 | 0.594 | 21.85 | +26.78 | 0.00 | +35.73 |
| prob-only (geo) | −0.148 | 0.444 | −13.88 | −8.95 | −35.73 | 0.00 |
| random | 0.086 | 0.533 | −4.93 | 0.00 | −26.78 | +8.95 |

On synthetic data TAP already beats random, reward-only, and probability-only
(`verdict: beat_random=True beat_reward=True beat_prob=True`). The ridge and
candidate-only learned baselines edge SmallTAP here; on the real collection this
is exactly the comparison the spec says to resolve (and to report the simpler
model as TAP v1 if it wins). These synthetic numbers will be replaced by the real
72-label run via `tap.run_all --parquet-dir outputs/tap/<run_id>/parquet`.

## 8. Wave 2 status — launcher hardening (done) + smoke (blocked on credential)

**CPU integration gate — PASS:** `py_compile` of all `math_loop/*.py` + `tap/*.py`;
`python -m unittest tests.test_tap_schema tests.test_tap_engine tests.test_tap`
(55 tests); `math_loop.tap_controller --dry-run`; and `tap.run_all` on
`outputs/tap_synth_72`.

**Launcher (`run_prime_rl_math_loop.py`) — hardened + tested (15 tests incl. a
SIGTERM-teardown subprocess test):** default provider `lambdalabs`; prime-rl
pinned via the **required** `--prime-rl-commit`; a **fail-closed** cost monitor
(`offer price/h × elapsed`) plus an **independent wall-clock deadline** that reap
the pod and exit non-zero on breach *or on the monitor's own failure*;
`pod_id.txt` on create; atexit + SIGINT/SIGTERM + monitor-breach teardown that
reaps by `tap-v1-smoke-` prefix **and** by id; `--keep-pod` forbidden;
`--gpu-count > 1` and any non-`--smoke` run gated behind `TAP_ALLOW_FULL_RUN=1`
(never set by the loop); SSH via `/workspace/private_key.pem` (chmod 600, verified
against the registered `prime-intellect` public key).

**`reap_pods.py`** lists/terminates pods by id or name prefix and refuses any
empty or non-`tap-v1-` prefix.

**Credential — RESOLVED:** a `PRIME_API_KEY` was supplied; `prime whoami`
authenticates and `prime availability list --gpu-type H100_80GB --output json`
returns a non-empty offers array (8 offers; 4 lambdalabs, incl. a 1×H100 at
\$3.29/h). The key lives only in the environment, never on disk in the repo.

**Plan unknown (d) — CONFIRMED at the pin (from source):** at prime-rl HEAD
`4d361ad`, `class AdvantageOutputs` exists in `prime_rl/orchestrator/advantage.py`
and `verifiers` is a declared dependency, so the not-feature-degradable pre-flight
import will not block.

**Smoke — BLOCKED on the collection driver:** the real (non `--dry-run`)
`math_loop.tap_controller.run_controller` is a stub (`raise SystemExit`), and the
per-candidate `_branch_worker` expects pre-generated rollouts/probes — the on-pod
rollout + per-token logprob/entropy extraction against the real prime-rl (plan
unknowns a/b/c) must be built before any labels exist. That file is read-only in
this wave and the work is substantial, so the smoke cannot produce a mini-Parquet
here; no pod was provisioned to re-confirm a known, documented gap. The launcher,
reaper, cost/wall-clock guards, schema, featurizer, and eval are all verified on
CPU + synthetic data and the credentialled provisioning path (auth, availability,
pin, import (d)) is confirmed. See `RUNBOOK.md` to complete the driver and run.

## 9. Reproduction (CPU)

```bash
python -m py_compile math_loop/*.py tap/*.py
python -m unittest tests.test_tap_schema tests.test_tap_engine tests.test_tap
python -m unittest tests.test_launcher_teardown
python -m math_loop.tap_controller --dry-run
VIRTUAL_ENV=$PWD/.venv uv run --no-project \
  python -m tap.run_all --parquet-dir outputs/tap_synth_72 --out outputs/tap_report
python run_prime_rl_math_loop.py --dry-run --smoke --prime-rl-commit <sha>
```
