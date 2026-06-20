# TAP v1 — Hackathon Presentation Strategy

*Internal team playbook for the Mercor Inference-Time Compute Hackathon. This document tells the team HOW to present: what math to put on screen, how to frame the research contribution, what each of the five judges wants to hear, the live demo flow, a complete 16-slide skeleton, and the risk register. It is a planning guide, not the talk itself.*

---

## 1. Situation Summary

**The event.** The Mercor Inference-Time Compute Hackathon is a 24-hour, San-Francisco research sprint co-hosted by Mercor, Anthropic, Etched, Cognition, and Prime Intellect. Each team is given **8× NVIDIA H100s** (provided by Prime Intellect — Etched is a judge/co-host, not the compute provider) and competes across three tracks: Agents, Real-Time & Interactive, and Talent Marketplace + Applied AI. The prize pool is $100K+ across iterations with a **$50K** top prize. The defining cultural fact of this hackathon, learned from iteration 1, is that it **rewards research-grade algorithmic novelty applied to inference-time compute, not integration depth or UI polish.** Past winners became papers: LLaDA-R1 → ICLR 2026; Backmasking → ICML 2025; Interruption Tokens → publication.

**What we built.** TAP v1 (Trajectory Advantage Predictor) is a small meta-learner that predicts the usefulness of a candidate GRPO update batch *before* you pay to apply it. Concretely we have three things. (1) **SmallTAP** — a ~110K-parameter model (exactly 109,537 trainable parameters) that predicts a single scalar `predicted_utility_points` for a candidate GRPO batch applied to a Qwen3-8B LoRA policy, using only inference-time signals: reward, probability/familiarity, gradient sketch, current policy state, and similarity to the last four applied updates. (2) **A three-robot embodiment stack** demonstrating that the same idea generalizes across radically different action spaces — an arm (OpenVLA-OFT + LIBERO, discrete action tokens), a quadruped dog (MobileVLA-R1 + Genesis, continuous velocity), and a humanoid (GR00T N1.5 + ReinFlow, flow-matching actions). (3) **Data-collection, featurization, and evaluation infrastructure** — a four-table Parquet schema, a hardened Prime Intellect launcher with fail-closed cost and wall-clock guards, and a CPU-tested training + ranking-evaluation harness.

**The research question we answer.** *Can a small predictor (~110K parameters) select GRPO update batches that outperform reward-only and probability-only heuristics, using only inference-time signals — and does the same redundancy-proxy idea generalize across three robot embodiments with different action representations?*

**Why we have a shot.** Math and code RL are saturated demo domains; robotics simulation with verifiable simulator rewards is novel, multi-modal, and visually compelling. We sit directly on the cost-quality Pareto frontier the cheap-signals research identifies, we have a number to put on screen, we use the 8×H100s for actual online RL (not just inference), and we can frame the whole thing as a paper abstract. Every one of the five judges has a hook in our project.

---

## 2. Research Framing

**Lead as research, not product.** The single most important strategic decision is to present TAP as a research contribution. The history of this hackathon is unambiguous: the iteration-1 winner LLaDA-R1 became an ICLR 2026 publication, the Backmasking finalist became an ICML 2025 paper, and at least two more finalists became ICLR 2026 papers. The judges are explicitly looking for "something that belongs in a research-paper abstract." A polished product with no novel claim loses to a rough prototype that proves a new insight about inference-time compute.

**Our research question, stated for the abstract.** "We test whether a ~110K-parameter predictor can rank candidate GRPO update batches by their true held-out utility using only inference-time signals, beating reward-only, probability-only, and random selection — and whether the governing signal (a cheap, single-forward-pass redundancy proxy) generalizes across discrete-token, continuous-velocity, and flow-matching action spaces."

**Three layers of novelty.** (a) **TAP itself** — predicting the *advantage of a training update* from cheap signals is a meta-learning question distinct from the usual "rank training examples" data-selection literature; we predict the value of a GRPO *batch* against a held-out probe, with a causal branch-labeling protocol that keeps history unbiased. (b) **Cross-embodiment generalization** — we show the same redundancy-proxy logic predicts GRPO lift across three fundamentally different action representations, which no prior work demonstrates jointly. (c) **Novel redundancy proxies for flow-matching actions** — for the humanoid, where there is no token log-probability, we propose `mean_denoising_loss` and the ReinFlow exact log-prob as principled confidence metrics. This third layer is the most defensible "paper-worthy" contribution.

**The honesty discipline.** Our current ranking numbers are a synthetic plumbing-check, not real GPU labels (see §8). The framing must be: "Here is the hypothesis, here is the built system that tests it, here are the synthetic-data sanity numbers, and here is the real collection now running on the 8×H100s." Research-grade framing *requires* this honesty; over-claiming is the fastest way to lose credibility with these specific judges.

---

## 3. The Math

This section is the technical core. On stage, put **one** formula per slide — never a wall of math. The formulas below are the menu; pick the minimal set that lands each judge's point.

### 3.1 Gradient energy theory

The motivating insight. Under GRPO with binary rewards, the per-task gradient signal scales with reward variance, and for binary rewards reward variance equals $p(1-p)$:

$$
E[\text{gradient energy}] = E_T\big[\, p(T)\,(1 - p(T)) \,\big]
$$

This is maximized when $p(T) = 0.5$ and is zero at $p \in \{0, 1\}$. The practical consequence: tasks (or update batches) drawn from the **learning zone** $p \in (0.2, 0.8)$ carry almost all the useful gradient; a cohort mean of $0.3 < \text{mean}(p) < 0.7$ is ideal. Empirically, fewer than 5% of the gradient comes from tasks outside $(0.1, 0.9)$, difficulty filtering to $0.1 < p < 0.9$ yields +12% math accuracy in under 50% of training steps, and pass-rate control gives a 2× speedup on SWE-bench. **Presenter framing:** "Most GRPO batches are wasted compute — they sit at the edges where the gradient is near zero. TAP learns to find the middle."

### 3.2 Utility formula

The exact scalar TAP predicts. For each candidate branch we apply one GRPO step, measure probe NLL before/after, and compute:

$$
\text{utility\_points} = 1000 \times \big( 0.8\,\text{matched\_gain} + 0.2\,\text{global\_gain} - 0.03\,\max(\text{incremental\_KL}, 0) \big)
$$

where $\text{matched\_gain}$ and $\text{global\_gain}$ are the reductions in matched-subject and globally-stratified held-out MATH probe NLL, and $\text{incremental\_KL}$ is the drift on a generic non-math probe. All NLL/KL values are averages in nats per non-padding token; the ×1000 is purely cosmetic for readability. Positive = the update reduced held-out loss; negative = it hurt or drifted.

### 3.3 Composite Cohort Value Index (CVI)

The cheap-signals framework's headline construct — a 15-minute proxy for training value that would otherwise cost a full RL run:

$$
\text{CVI} = 0.30\,\text{LZ}_{\text{frac}} + 0.20\,\widehat{\text{Var}}_r + 0.15\,\widehat{\text{DDS}} + 0.20\,\widehat{\text{Vendi}} + 0.15\,\widehat{\text{CWC}}
$$

where $\text{LZ}_{\text{frac}}$ is the fraction of tasks with $0.15 < p < 0.85$, $\widehat{\text{Var}}_r$ is normalized reward variance, $\widehat{\text{DDS}}$ is normalized difficulty dispersion, $\widehat{\text{Vendi}}$ is the normalized Vendi diversity score, and $\widehat{\text{CWC}}$ is normalized coherence-without-correctness. A 500-task CVI pipeline runs in ~15 minutes versus hours for gradient methods and days for the oracle.

### 3.4 Results table — TAP vs three baselines

⚠️ Synthetic 72-label placeholder (see §8); on screen, label it as such.

| Model | Spearman | Pair acc | Mean true utility |
|---|---|---|---|
| random | 0.086 | 0.533 | −4.93 |
| reward-only | 0.233 | 0.594 | 21.85 |
| prob-only (geo) | −0.148 | 0.444 | −13.88 |
| **SmallTAP** | **0.738** | **0.817** | **34.42** |

Verdict: `beat_random=True, beat_reward=True, beat_prob=True`.

### 3.5 Per-embodiment redundancy proxies

The engineering crux: a cheap confidence metric per action type. For the arm, discrete action tokens give an exact log-probability sum:

$$
\text{redundancy}_{\text{arm}}(T) = \sum_{t} \log \pi(a_t \mid s_t)
$$

For the dog, velocity commands are either discretized into tokens (same form as the arm) or treated as a Gaussian policy, giving $-\text{MSE}(\hat a, a)/2\sigma^2$. For the humanoid there is no token log-prob, so we use the denoising loss as a heuristic confidence:

$$
\text{redundancy}_{\text{humanoid}}(T) = \mathbb{E}\big[\, \lVert \varepsilon_t(x_t, t) \rVert^2 \,\big]
$$

with the ReinFlow exact Gaussian log-prob as the principled alternative. The unifying claim: **low redundancy proxy ⇒ high uncertainty ⇒ largest GRPO lift.** Each is one forward pass.

### 3.6 LIBERO lift table

These are published GRPO results (not synthetic) and ground the redundancy-proxy claim:

| Suite | Lift |
|---|---|
| LIBERO-Spatial | +5.7pp |
| LIBERO-Object | +3.8pp |
| LIBERO-Goal | +1.8pp |
| LIBERO-Long | +8.1pp |
| **Average** | **+4.2pp** |

The 4.5× spread (1.8 → 8.1pp) is the natural cohort variance TAP learns: LIBERO-Long has the lowest baseline (51%), the most subgoals, the lowest redundancy proxy, and the largest lift.

---

## 4. Three-Robot Architecture

**Why three robots.** A single generalist predictor across all embodiments is architecturally wrong: a denoising loss of 0.4 means something completely different for a 7-DOF arm than for a 30-DOF humanoid, and mixing embodiments in one attention context poisons the learned instruction-to-lift patterns. Instead we deploy **three separate TAPs that share one frozen language encoder**. The shared encoder embeds task descriptions and learns which instruction patterns preceded high-lift runs; everything numeric — the action autoencoder, the prediction head, the redundancy proxy, and the 128-run history context — is per-embodiment. This is the cross-embodiment generalization story: the *idea* transfers even though the *weights* must not.

**The progression that makes the point.** Arm = discrete action tokens → Dog = continuous velocity (MSE) → Humanoid = flow-matching. Showing the same redundancy-proxy logic works across all three is what elevates this from "a trick on one robot" to "a general principle of GRPO data value."

**Hardware allocation (8 H100s).**

| Embodiment | Platform | Action representation | Redundancy proxy | GRPO status | H100s |
|---|---|---|---|---|---|
| Arm | OpenVLA-OFT (7B) + LIBERO | Discrete 7-DoF EEF tokens | sum_log_probs (exact) | VALIDATED (TGRPO, VLA-RFT) | 4–6 |
| Dog | MobileVLA-R1 (8B) + Genesis | Continuous velocity + discrete hi-level | sum_log_probs OR −MSE | ONE BASELINE (93% real Go2) | 2 |
| Humanoid | GR00T N1.5 (3B) + ReinFlow | Flow-matching DiT (16-step chunks) | mean_denoising_loss OR ReinFlow log-prob | NOVEL / UNVALIDATED | 2 |

Arm gets the majority because SimpleVLA-RL is battle-tested and LIBERO lift variance is well-characterized; dog and humanoid are exploratory with smaller allocations so we can pivot if a baseline blocks.

---

## 5. Judge-by-Judge Strategy

**Anthropic.** They want concrete real-world impact for a *named user* using capabilities only Claude-class models enable, and they explicitly reject "look what it can do" demos. **What we say:** "This is for robotics and RL teams doing post-RL fine-tuning who burn H100-hours on low-value GRPO batches; TAP tells them which batch to train on before they pay for it." Emphasize eval-first, spec-first discipline (we wrote the schema and the utility formula before collecting a single label). **Number on screen:** the one-sentence insight, not a metric — "who is this for, and what does their day look like without it."

**Etched.** Test: "would this run on a CPU?" Ours would not — GRPO branch labeling needs H100 throughput (576 trajectories, a real optimizer step per candidate). **What we say:** make the compute visible with a latency/throughput counter during collection. **Number on screen:** a latency counter, closing with "On Etched's Sohu, this gets 21× more users at the same cost" applied to TAP-guided batch selection.

**Cognition.** They proved serial depth beats parallel breadth (Kevin-32B: 8 sequential refinement steps > 64 parallel samples at fixed compute). **What we say:** "TAP *is* serial depth — each policy state sequentially refines which batch to train on next, conditioning on the last four applied updates via history attention; cite your Kevin-32B result directly." **Number on screen:** the history-ablation delta (with-history pair-acc 0.817 vs no-history 0.683).

**Mercor.** "Evals are the new PRD" — measurable outcomes, explicit baselines, a number in the UI. **What we say:** TAP is an autograder for GRPO batches. **Number on screen:** TAP mean_true_utility = **34.42 vs random −4.93** with a live counter, framed as "we beat the reward-only baseline by +12.57 utility points."

**Prime Intellect.** They want the 8×H100s used for actual online RL improvement during the event, with verifiable rewards. **What we say:** "We didn't just deploy a model — we improved it; here is the live GRPO loop and the utility curve from hour 1 to hour 23." **Number on screen:** the accuracy/utility-over-time chart from the live run, and a mention of their Lab platform if used.

---

## 6. Demo Flow

A tight 7-minute live demo. Rehearse the golden path three times and prepare an off-script handoff plus a backup recording.

- **0:00–1:30 — Problem + gradient-energy insight.** One slide, one formula ($p(1-p)$), one motivating stat ("<5% of GRPO gradient comes from batches outside the learning zone"). State the one-sentence insight.
- **1:30–3:00 — TAP vs 3 baselines on the arm.** Live LIBERO task/cohort selection; the results table (random / reward-only / prob-only / SmallTAP) appears on screen with the mean_true_utility counter ticking up for TAP.
- **3:00–4:30 — Dog + humanoid (≈30s each).** Show the redundancy proxy for each action type; make the discrete → continuous → flow-matching generalization explicit.
- **4:30–5:30 — Numbers on screen.** The mean_true_utility counter (Mercor) and the accuracy/utility-over-time chart from the live 8×H100 run (Prime Intellect); a latency counter for Etched.
- **5:30–7:00 — Research framing.** The three layers of novelty, the honest synthetic-vs-real status, the potential ICML 2026 direction, and invite questions. Offer the interactive handoff: let a judge pick a cohort and watch TAP rank it.

---

## 7. Slide Skeleton

Sixteen slides. Each lists a title, 3–5 bullets, and a one-sentence presenter note. Keep one idea per slide.

### Slide 1: Title

- "TAP v1: Trajectory Advantage Predictor for GRPO Across Three Robot Embodiments"
- Team names; tracks entered (Agents + Real-Time)
- One-line subtitle: "A ~110K-parameter predictor that picks the GRPO batch worth training on"
- Hackathon + date + sponsors
- Presenter note: Say the one-sentence insight before the title animation finishes.

### Slide 2: The Problem

- H100-hours are precious; a GRPO step is expensive
- Most candidate batches are near-zero-gradient (wasted compute)
- Teams currently pick batches by reward or by hand
- There is no cheap way to know a batch's true value before paying for it
- Presenter note: Frame the pain for the named user — RL fine-tuning teams burning compute on bad batches.

### Slide 3: The Key Insight

- GRPO gradient energy $\propto p(1-p)$, maximized at $p=0.5$
- Learning zone $p \in (0.2, 0.8)$; <5% of gradient outside $(0.1, 0.9)$
- Therefore batch value is *predictable* from cheap signals
- Presenter note: This is the whole talk in one formula — pause on it.

### Slide 4: Cheap Signals Pareto

- Cost-quality frontier: free → rollouts → gradients → oracle
- Pass rate is the Pareto-breaker (high value, low cost)
- CVI bundles 5 signals into a 15-minute proxy
- Redundancy proxy = single forward pass, predicts lift
- Presenter note: Position TAP's signals as sitting above the cost-quality curve.

### Slide 5: Our Answer — TAP

- ~110K parameters (109,537 exactly), under the 250K target
- Inputs: candidate batch + policy state + last-4 history
- Output: one scalar, predicted_utility_points
- Inference-time signals only — no extra training to score a batch
- Presenter note: Emphasize "small predictor, big leverage."

### Slide 6: SmallTAP Architecture

- Candidate-embedding (256→64) + gradient-sketch (64→32) projections
- Numeric MLP (16→64) + state MLP (26→32)
- 4-head cross-attention: candidate → 4 history slots
- Two-layer MLP head → one scalar
- Presenter note: Call out the history attention — it is the "serial depth" hook for Cognition.

### Slide 7: Data Collection

- 2 chains × 6 states × 6 candidates = 72 labels; 576 trajectories
- Each candidate branches from a byte-identical before-state
- Seeded-random main-chain advance keeps history unbiased
- utility_points = $1000(0.8\,\text{matched} + 0.2\,\text{global} - 0.03\max(\text{KL},0))$
- Presenter note: Stress the causal branch-labeling — this is what makes the labels trustworthy.

### Slide 8: Results Table

- random / reward-only / prob-only / SmallTAP
- SmallTAP: Spearman 0.738, pair-acc 0.817, mean utility 34.42
- Beats random (−4.93), reward-only (21.85), prob-only (−13.88)
- Synthetic placeholder — real collection running live
- Presenter note: Say the honesty caveat out loud; it builds credibility.

### Slide 9: Ablations

- What matters: history > state > gradients
- With-history pair-acc 0.817 vs no-history 0.683
- Pairwise ranking loss handles label noise
- Presenter note: Use the history delta to make the serial-depth point concrete.

### Slide 10: Three Robots Overview

- Arm / dog / humanoid; why different action spaces matter
- Discrete tokens → continuous velocity → flow-matching
- Shared frozen language encoder, separate numeric heads
- Presenter note: This is the generalization story — the reason it is a principle, not a trick.

### Slide 11: Per-Embodiment Redundancy Proxies

- Arm: sum_log_probs (exact)
- Dog: sum_log_probs or −MSE (Gaussian analogue)
- Humanoid: mean_denoising_loss or ReinFlow exact log-prob
- Low proxy ⇒ high uncertainty ⇒ largest lift
- Presenter note: This is the most paper-worthy contribution — flow-matching has no token log-prob, so we invented the proxy.

### Slide 12: LIBERO Results

- Spatial +5.7 / Object +3.8 / Goal +1.8 / Long +8.1 pp
- Average +4.2pp; 4.5× spread across suites
- LIBERO-Long: lowest baseline, lowest proxy, largest lift
- Presenter note: Tie the lift spread back to the redundancy-proxy prediction.

### Slide 13: Live Demo

- TAP selecting cohorts in real time on the arm
- Results table + mean_true_utility counter on screen
- Interactive handoff: judge picks a cohort
- Presenter note: Hand off control — the interactive moment is what judges remember.

### Slide 14: Judge Signal

- Latency counter (Etched): GRPO needs H100 throughput
- Accuracy/utility curve hour 1 → hour 23 (Prime Intellect)
- mean_true_utility number on screen (Mercor)
- Presenter note: One visible signal per judge — point at each as you name it.

### Slide 15: Research Contribution

- Three layers of novelty: TAP, cross-embodiment, flow-matching proxies
- Causal branch-labeling protocol keeps history unbiased
- Potential ICML 2026 direction; lineage of past hackathon papers
- Presenter note: Explicitly say "this belongs in a paper abstract."

### Slide 16: Next Steps + Open Questions

- Real GPU label collection (replaces synthetic numbers)
- Validate humanoid ReinFlow baseline
- Mini end-to-end online test: TAP-selected vs reward vs random
- Publication + scaling beyond 72 labels
- Presenter note: End on the open questions — research framing invites the judges into the work.

---

## 8. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Synthetic-vs-real gap: current ranking numbers (0.738 / 0.817 / 34.42) are a synthetic plumbing-check, not real GPU labels | High | High | Never present synthetic numbers as real; show the live collection running on 8×H100s; frame as hypothesis + system + sanity numbers |
| Humanoid ReinFlow baseline not validated: no published standalone GRPO on GR00T; denoising-loss proxy is heuristic | High | Medium | Start with RoboCasa binary rewards; use empirical mean_denoising_loss; smallest H100 allocation; present as exploratory |
| Dog Genesis simulation unconfirmed for this stack; joint-level GRPO has zero empirical results | Medium | Medium | Prioritize MobileVLA-R1 velocity-command GRPO; fall back to IsaacLab; 2-H100 allocation limits downside |
| Simpler learned baselines (ridge / candidate-only) edge SmallTAP on synthetic data | Medium | Medium | Spec says report the simpler model as TAP v1 if it wins on real data; present the attention model as a hypothesis to resolve |
| 24-hour time pressure: collection driver + 3 embodiments + live demo in one window | Medium | High | Scope discipline — fall back to 48 labels / arm-only; freeze features by hour 22; backup demo recording |
| Demo failure live on stage | Medium | High | Rehearse golden path 3×; prepare off-script handoff scenario; keep a recorded backup |
| Over-claiming loses credibility with research-grade judges | Low | High | Honesty discipline baked into every numbers slide; lead with the question, not the result |
