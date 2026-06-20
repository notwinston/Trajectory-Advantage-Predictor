# Hackathon Presentation Strategy — Master Extraction Notes

*Internal merge of all source material for the Mercor Inference-Time Compute Hackathon presentation. Compiled by the main agent from three Explore subagent files plus two primary sources.*

**Sources merged here:**
- `cheap_signals.md` (`docs/Inference_Time_Cheap_Signals/`) → distilled in `NOTES_signals.md`
- `hackathon-intel.md` (`docs/Inference_Time_Hackathon_Intel/`) → distilled in `NOTES_intel.md`
- `robotics_tap_full.md` (`docs/Robotics_LLM_Mercor/`) → distilled in `NOTES_robots.md`
- `Implementation_Report/report.md` (the actual built TAP v1 system)
- `TAP_v1_3_4_hour_plan.txt` (the compressed 3–4 hour execution plan)

---

## 1. KEY NUMBERS (memorize these — they go on screen)

### SmallTAP within-state ranking results (⚠️ SYNTHETIC 72-label placeholder, NOT real GPU labels yet)

| Model | Spearman | Pair acc | Mean true utility | Lift vs random | Lift vs reward | Lift vs prob |
|---|---|---|---|---|---|---|
| **TAP (SmallTAP)** | **0.738** | **0.817** | **34.42** | **+39.35** | **+12.57** | **+48.30** |
| ridge | 0.829 | 0.861 | 35.79 | +40.72 | +13.94 | +49.67 |
| candidate-only | 0.843 | 0.872 | 36.59 | +41.52 | +14.74 | +50.46 |
| no-history MLP | 0.481 | 0.683 | 32.92 | +37.85 | +11.07 | +46.80 |
| reward-only | 0.233 | 0.594 | **21.85** | +26.78 | 0.00 | +35.73 |
| prob-only (geo) | −0.148 | 0.444 | **−13.88** | −8.95 | −35.73 | 0.00 |
| random | 0.086 | 0.533 | **−4.93** | 0.00 | −26.78 | +8.95 |

**The headline comparison:** TAP mean_true_utility = **34.42** vs reward-only **21.85** vs prob-only **−13.88** vs random **−4.93**. Verdict from `tap.run_all`: `beat_random=True beat_reward=True beat_prob=True`.

⚠️ **HONESTY NOTE (from `Implementation_Report/report.md` §7):** these numbers are the synthetic plumbing-check on `outputs/tap_synth_72`, *not* a scientific result. GPU collection was blocked on the Prime Intellect credential + collection driver; the latest commit ("Implement the TAP collection driver") closes that gap. On the real collection, the spec says to report the *simpler* learned model as TAP v1 if it wins (ridge/candidate-only currently edge SmallTAP on synthetic data). **Do not claim these as real robot/MATH numbers on stage.**

### LIBERO lift table (arm, from `robotics_tap_full.md` — these ARE published GRPO results)

| Suite | SFT baseline | GRPO | Lift | Redundancy |
|---|---|---|---|---|
| LIBERO-Spatial | 84.7% | 90.4% | **+5.7pp** | Medium |
| LIBERO-Object | 88.4% | 92.2% | **+3.8pp** | Medium |
| LIBERO-Goal | 79.2% | 81.0% | **+1.8pp** | High (mostly solved) |
| LIBERO-Long | 51.1% | 59.2% | **+8.1pp** | Low (uncertain, multi-step) |
| **Average** | 76.5% | 80.7% | **+4.2pp** | — |

The 4.5× spread (1.8 → 8.1pp) across suites is the natural cohort-variance signal TAP learns. SimpleVLA-RL SOTA: 97.6% on LIBERO-Long; cold-start 17.3% → 91.7% (+74.4pp) with 1 trajectory/task.

### System scale numbers
- **SmallTAP trainable parameters: 109,537** (~110K, under the 250K target).
- Data collection: **2 chains × 6 states × 6 candidates = 72 labels**; 72 × 8 = 576 trajectories.
- Policy: Qwen3-8B, LoRA rank 16, GRPO, BF16, non-thinking, 192-token max completion, 4 completions × 2 prompts = 8 trajectories per candidate.
- Compute: 4–5 H100s for collection (H100 0 = main chain/rollouts; H100 1–4 = parallel branches/probes).

---

## 2. KEY FORMULAS

### Utility label (the prediction target — `TAP_v1_3_4_hour_plan.txt` + `Implementation_Report/report.md`)
```
matched_gain           = matched_probe_nll_before − matched_probe_nll_after
global_gain            = global_probe_nll_before  − global_probe_nll_after
incremental_generic_kl = generic_kl_after − generic_kl_before
utility_points         = 1000 × (0.8·matched_gain + 0.2·global_gain
                                  − 0.03·max(incremental_generic_kl, 0))
```
All NLL/KL are averages in nats per non-padding token. Positive = reduced held-out MATH loss; the ×1000 is cosmetic.

### Gradient energy ∝ p(1−p) (from `cheap_signals.md`)
```
E[gradient energy] = E_T[ p(T) · (1 − p(T)) ]      (binary rewards: Var[r] = p(1−p))
```
Maximized at p = 0.5; zero at p ∈ {0,1}. Optimal per-task zone **p ∈ (0.2, 0.8)**; cohort mean 0.3 < mean(p) < 0.7. Empirical: difficulty filter 0.1<p<0.9 → +12% math in <50% steps; pass-rate control → 2× speedup on SWE-bench; <5% of gradient comes from tasks outside (0.1, 0.9).

### Composite Cohort Value Index (CVI) (from `cheap_signals.md`)
```
CVI = 0.30 × learning_zone_fraction
    + 0.20 × reward_variance (normalized: min(mean_var × 4, 1))
    + 0.15 × difficulty_dispersion (normalized: min(dds × 4, 1))
    + 0.20 × vendi_score (normalized: min(vendi / n_tasks, 1))
    + 0.15 × coherence_without_correctness (normalized: min(mean_cwc × 10, 1))
```
learning_zone_fraction = fraction of tasks with 0.15<p<0.85. 500-task CVI pipeline ≈ 15 min vs hours (gradients) vs days (oracle).

### Per-embodiment redundancy proxies (from `robotics_tap_full.md`)
- **ARM** (discrete action tokens): `sum_of_predicted_tokens_probs` = Σ log π(a_t). Exact, 1 forward pass.
- **DOG** (continuous velocity): `sum_of_log_probs` if velocity discretized to tokens, OR `−MSE(pred, target)` for Gaussian-policy analogue. 1 forward pass.
- **HUMANOID** (flow-matching): `mean_denoising_loss` = E[‖ε_t(x_t,t)‖²] (empirical heuristic, validated by Diff-DAgger) OR **ReinFlow exact log-prob** (exact, requires noise-injection adapter) OR GMM on denoising vectors.

**Core thesis:** one cheap, single-forward-pass confidence metric that predicts GRPO lift across discrete-token → continuous-MSE → flow-matching action spaces. Low redundancy proxy → high uncertainty → largest GRPO lift (LIBERO-Long is the canonical example).

---

## 3. JUDGE THESES (one paragraph each — from `hackathon-intel.md`)

**Anthropic (Claude).** Rejects "look what it can do" demos; wants agentic systems solving concrete problems for *named users* that could only be built with Claude's extended thinking / multi-agent / long-context. Eval-first, spec-first (4+ hrs planning before code). Winning one-sentence test: "Who is this for, and what does their day look like without it?" → **For us:** robotics/RL teams doing post-RL fine-tuning who waste H100-hours on low-value GRPO batches. Domain expertise beats coding flash (doctors/attorneys/teachers win).

**Etched (Sohu chip).** Wants use cases economically infeasible on H100s but viable at Sohu's 21× throughput / 10× cheaper cost. Test: "Would this run on a CPU? If yes, it won't impress." Make speed *visible* — put a latency/FPS counter in the UI. → **For us:** GRPO branch labeling needs H100 throughput (576 trajectories, per-candidate optimizer steps); show a latency counter; close with "21× more users at the same cost" applied to TAP-guided batch selection.

**Cognition (Devin / Kevin-32B).** Algorithmic novelty in inference-time compute *allocation*. Their result: 8 sequential refinement steps with executable feedback beats 64 parallel samples at fixed compute — "serial depth beats parallel breadth." Wants verifiable/executable feedback loops, quantitative benchmarks. → **For us:** TAP IS serial depth — each state sequentially refines which batch to train on next, conditioning on the last-4 applied updates (history attention). Cite Kevin-32B explicitly.

**Mercor (APEX / Monty / Era of Evals).** "Evals are the new PRD." Wants measurable outcomes with explicit baselines and a number in the demo UI. Monty = 10,000 interviews/day at 700ms. → **For us:** put TAP mean_true_utility = **34.42 vs random −4.93** on screen with a live counter; frame as an autograder for GRPO batches. Beat a baseline and *say the number*.

**Prime Intellect (distributed training / INTELLECT / Lab).** Wants 8×H100s used in architecturally interesting ways — online RL fine-tuning *during* the hackathon, RL environments with verifiable rewards, actual model improvement. Launched Lab (May 2026) for exactly this. → **For us:** show the accuracy/utility curve from hour 1 to hour 23; show real GRPO training happening live on the 8×H100s; "we didn't just deploy a model — we improved it." They provide the compute.

---

## 4. THREE WINNING STRATEGY PATTERNS (from `hackathon-intel.md`)

1. **Real-time interactive world model (20+ FPS, transformer-only).** Oasis-style (DiT+ViT, persistent KV cache). Demo moment: hand judge the keyboard, FPS counter at 22, disconnect H100s → drops to 6, reconnect → 22. Appeals to Etched + Prime Intellect + Anthropic.
2. **Serial-refinement agent with PRM scoring.** 8 sequential steps beat 64 parallel samples at fixed compute, on a verifiable domain (FrontierCode / MATH). Demo moment: PRM 0.42 → 0.84 over 8 steps; GPT-4 baseline failed at step 3.
3. **Sub-300ms voice AI with visible latency dashboard.** Beat Monty's 700ms; hand judge the mic. Close: "Monty 700ms, us 280ms, Sohu sub-50ms."

**Universal patterns:** algorithmic novelty over polished scaffolding ("does it belong in a paper abstract?"); speed as a UX feature; the interactive handoff; verifiable outcomes with quantitative claims; the one-sentence insight written before any code.

**Past winners / publication track record:** LLaDA-R1 (iteration 1, $40K) → ICLR 2026; Backmasking (PRM for diffusion LMs, ~2× GSM8K) → ICML 2025; Interruption Tokens (34.5% refusal-rate improvement, 3.5% accuracy loss) → publication; two more finalists → ICLR 2026. **Lesson: this hackathon rewards research-grade algorithmic novelty applied to inference-time compute, not integration depth or UI polish.**

**Logistics:** SF, 24-hour format. Prize pool $100K+ across iterations; first place $50K. 8×NVIDIA H100s per team provided by Prime Intellect (Etched is judge/co-host, not the compute provider). Tracks: (1) Agents, (2) Real-Time & Interactive, (3) Talent Marketplace + Applied AI.

---

## 5. THREE-ROBOT ARCHITECTURE (from `robotics_tap_full.md`)

Shared frozen-LLM language encoder over task descriptions → MHA learns which instruction patterns preceded high-lift runs. **Separate per-embodiment** numeric autoencoder, prediction head, redundancy proxy, and 128-run context (mixing embodiments poisons attention — a denoising loss of 0.4 means different things for a 7-DOF arm vs a 30-DOF humanoid).

| Embodiment | Platform | Action representation | Redundancy proxy | GRPO status | H100s |
|---|---|---|---|---|---|
| **Arm** | OpenVLA-OFT (7B) + LIBERO | Discrete 7-DoF EEF token deltas (256 bins) | sum_log_probs (exact) | **VALIDATED** (TGRPO, VLA-RFT) | 4–6 |
| **Dog** | MobileVLA-R1 (8B LLaMA3+LoRA) + Genesis | Continuous velocity [Vx,Vy,ω,α] + discrete hi-level | sum_log_probs OR −MSE | **ONE BASELINE** (MobileVLA-R1, 93% real Go2) | 2 |
| **Humanoid** | GR00T N1.5 (3B) + ReinFlow | Flow-matching DiT, K=4 Euler, 16-step chunks | mean_denoising_loss OR ReinFlow log-prob | **NOVEL/UNVALIDATED** | 2 |

Total = 8 H100s. Arm dominates because SimpleVLA-RL is battle-tested. The progression discrete → continuous → flow-matching is the generalization story.

---

## 6. RISK ITEMS

- **Synthetic vs real gap (HIGH):** all current TAP ranking numbers (0.738 / 0.817 / 34.42) are synthetic plumbing-check placeholders per `Implementation_Report/report.md`; real GPU labels were blocked and only just unblocked by the collection driver. Never present synthetic numbers as real.
- **Humanoid ReinFlow baseline UNVALIDATED (HIGH):** no published standalone GRPO on GR00T; ReinFlow (NeurIPS 2025) not deployed at scale; mean_denoising_loss is a heuristic, no log-prob equivalence proven. Mitigation: start with RoboCasa binary rewards; empirical proxy.
- **Dog Genesis simulation UNCONFIRMED (MEDIUM):** Genesis is fast (430K× real-time) but not confirmed to match Go2 physics for this stack; joint-level GRPO has zero empirical results. Mitigation: prioritize MobileVLA-R1 velocity-command GRPO; fall back to IsaacLab.
- **Simpler baselines may win (MEDIUM):** ridge/candidate-only edge SmallTAP on synthetic data; spec says report the simpler model as TAP v1 if it wins on real data. Frame the attention model as a hypothesis, not a foregone conclusion.
- **24-hour time pressure (MEDIUM):** collection driver, 3 embodiments, and live demo in one window; scope discipline (fall back to 48 labels / arm-only if needed).

---

## 7. DEMO & FRAMING ANGLES (synthesized)

Lead as **research**: the question is "Can a ~110K-param predictor select GRPO update batches that beat reward-only and prob-only heuristics using only inference-time signals — and does it generalize across three robot action spaces?" Three layers of novelty: (a) TAP itself; (b) cross-embodiment generalization (discrete → continuous → flow-matching); (c) novel redundancy proxies for flow-matching actions. The cheap-signal story (`cheap_signals.md` Pareto frontier) plus the robot story (`robotics_tap_full.md` LIBERO lifts) plus the built system (`Implementation_Report/report.md`, `TAP_v1_3_4_hour_plan.txt`) is the full arc.
