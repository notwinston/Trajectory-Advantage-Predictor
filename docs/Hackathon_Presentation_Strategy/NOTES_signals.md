# Cheap Signals for Post-RL Data Cohort Value: Hackathon Presentation Notes

*Internal strategy notes | June 2026*

---

## KEY FORMULAS

### Lift (Oracle Metric)
```
lift(cohort, M) = accuracy(M_after_RL(cohort)) − accuracy(M_before) on held-out benchmark
```
The metric we want to predict cheaply without running full training.

### Pass Rate
```
p(T) = (1/N) Σ r(V(M(Tᵢ)))
```
Mean reward across N rollouts. **Most predictive cheap signal for RL.**

### Reward Variance (Binary Rewards)
```
Var_r(T) = (1/N) Σ (rᵢ - p(T))²
For binary rewards: Var[r] = p(1-p)
```
For non-binary rewards, variance provides additional signal beyond pass rate.

### Gradient Energy ∝ p(1−p) Theory
```
E[gradient energy] = E_T[p(T) · (1 - p(T))]
```
**Maximized when p(T) = 0.5 for all tasks.**
- Optimal range for individual tasks: **p ∈ (0.2, 0.8)**
- Cohort mean: **0.3 < mean(p) < 0.7**
- Maximum gradient energy at p = 0.5; zero gradient at p ∈ {0, 1}

### GRPO Gradient Per Task
```
g(T) = (1/N) Σᵢ (rᵢ - r̄) ∇θ log π(yᵢ | T)
```
Where r̄ = mean reward. Advantage weighting (rᵢ - r̄) determines gradient magnitude.

### Composite Cohort Value Index (CVI)
```
CVI = 0.30 × LZ_fraction 
    + 0.20 × min(mean_var × 4, 1.0)
    + 0.15 × min(dds × 4, 1.0)
    + 0.20 × vendi_normalized
    + 0.15 × cwc_normalized

where:
- LZ_fraction = fraction of tasks with 0.15 < p < 0.85
- mean_var = mean reward variance across tasks
- dds = variance of per-task pass rates (difficulty dispersion)
- vendi_normalized = min(vendi_score / n_tasks, 1.0)
- cwc_normalized = min(mean_cwc × 10, 1.0)
```

---

## COST-QUALITY PARETO FRONTIER

### Cost Tiers (from cheapest to most expensive)

| Tier | Cost | Examples |
|------|------|----------|
| **≈ Free** | No inference | Token length, SimHash dedup, perplexity |
| **N Rollouts** | N × forward pass | Pass rate, reward variance, policy entropy, trajectory diversity |
| **Embeddings** | O(n) encoder passes | Vendi Score, task diversity coefficient |
| **Gradients** | ≤1 backward pass/task | EL2N, TRAK, DataInf, gradient alignment |
| **Proxy Training** | Small model training | RHO-Loss, proxy RL lift |
| **Full Training** | Complete RL run | Oracle (true lift measurement) |

### Top Pareto-Breakers (Best Value per Compute Unit)

1. **Pass rate** — highest value per unit compute in RL setting (N rollouts → theoretical lower bound on gradient energy)
2. **Reward variance** — zero additional cost once rollouts computed
3. **Difficulty Dispersion Score (DDS)** — zero additional cost (second-order statistic from pass rates)
4. **SimHash deduplication** — free, provides 5–15% effective capacity recovery
5. **Vendi Score** — affordable embedding computation, catches low-diversity cohorts

---

## CHEAP SIGNALS ORGANIZED BY TIER

### Tier 1: Model-Agnostic (≈ Free)

- **Token/character length**: Weak predictor alone; use as hard filter only
- **Syntactic complexity**: Weak correlation with lift; useful for domain diversity assurance
- **Semantic metadata**: Domain classification, topic clustering (for diversity, not lift prediction)
- **SimHash deduplication**: O(n), catches trivial redundancy, recovers 5–15% capacity when near-dup fraction > 30%

**Practical use**: Hard filters only. Do not rely for lift prediction.

### Tier 2: Model-Dependent (N Rollouts, High Value)

#### Pass Rate (THE Primary Signal)
```
p(T) = (1/N) Σ r(V(M(Tᵢ)))
```
- **Theoretical basis**: Under GRPO, gradient signal ∝ Var[r] = p(1-p) for binary rewards
- **Key result** [arxiv:2504.03380]: Difficulty filtering to 0.1 < p < 0.9 achieves **+12% on math reasoning in <50% of training steps**
- **Key result** [arxiv:2605.05112]: Controlling pass rate distribution → **2x speedup on SWE-bench**
- **Optimal range**: 0.2 < p(T) < 0.8 per task; cohort mean 0.3 < mean(p) < 0.7
- **Cost**: N=20 is reliable; N=5 suffices for screening. N=20 → 5–15 min on A100 for 500-task cohort

#### Reward Variance
```
Var_r(T) = (1/N) Σ (rᵢ - p(T))²
```
- For binary rewards, algebraically equivalent to pass rate
- For partial-credit rewards, captures additional signal (e.g., bimodal vs. clustered)
- **Cost**: Zero additional cost once rollouts computed

#### Policy Entropy H(M|T)
- Measured at critical decision tokens
- Low = overconfident (memorized pattern); High = confused (OOD); Moderate = ideal learning zone
- **Cost**: Minimal (<5% overhead to rollout computation)

#### Trajectory/Rollout Diversity
- Mean pairwise edit distance among reasoning chains; prefix deduplication ratio
- If all N rollouts identical → single obvious approach → less room for RL to discover strategies
- **Cost**: O(N²) comparisons, <0.1 sec for N=10

#### Branching Factor B(T)
- Counts distinct valid solution paths (unique correct-rollout prefixes)
- B=1: memorization; B>3: generalization-rich
- **Cost**: Negligible

#### Max Score max_r(T) = max{rᵢ}
- Solvability gate: max_r(T) = 0 → unsolvable, exclude
- Prime learning zone: max_r(T) > 0 AND p(T) < 0.9
- **Cost**: Implicit in rollout collection

### Tier 3: Embeddings (Affordable Diversity Signals)

#### Vendi Score VS(S)
```
VS(S) = exp(H(K/n))
where K = cosine similarity kernel matrix (n×n)
and H = von Neumann entropy

Effective number of distinct elements in cohort.
If all identical: VS=1. If maximally distinct: VS=n.
Quality threshold: VS/n > 0.3
```
- **Cost**: O(n³) eigendecomposition; <60 sec CPU for n=1000
- **Evidence**: Low-Vendi cohorts → entropy collapse in RL training

#### Task Diversity Coefficient DC(S)
```
DC(S) = (2/n(n-1)) Σᵢ<ⱼ d(embed(Tᵢ), embed(Tⱼ))
Quality threshold: DC > 0.4
```
- Mean pairwise embedding distance
- **Cost**: O(n²), <1 sec for n=100

#### Embedding k-center Coverage
- Greedy selection to maximize minimum distance to selected set
- Useful as both metric and active selection algorithm

### Tier 4: Gradient-Based (Medium Cost, High Predictive Power)

#### EL2N (Error L2-Norm)
```
EL2N(x) = E[||f(x) - y||²] averaged over early training checkpoints
```
- Harder examples → more learning potential
- Pruning 30–50% of data using EL2N matches full-dataset performance
- **Cost**: ~10 training epochs (approximated by rollout error variance)

#### TRAK (Attributing Model Behavior)
```
Score(task) = projection(grad(task)) · projection(grad(val_target))
```
- Estimates influence: which training tasks most improve held-out performance
- **Cost**: Single backward pass per task; highly predictive

#### DataInf
```
Influence function approximation for LoRA-finetuned models
Pearson r = 0.97 with ground-truth influence
```
- Closed-form Hessian inverses; ~100x faster than full influence functions
- **Cost**: ~100x lower than traditional influence methods

#### Gradient Alignment
```
cosine_similarity(grad(task), grad(val_target))
```
- Does a task's gradient point toward the target benchmark?
- **Cost**: Two backward passes; strong first-pass predictor

### Tier 5: Proxy Training (Most Expensive Cheap Signal)

#### RHO-Loss (Reducible Holdout Loss)
```
RHO(x) = L_reference(x) - L_proxy(x)
```
- Measures what model can still learn: "learning frontier"
- **Cost**: Small proxy model training (one-time cost)
- **Key finding**: 6% of Alpaca (3K/52K) selected by RHO-based criteria matches full 52K performance

#### Proxy Model RL Lift Prediction
```
1.3B proxy model's RL lift → predicts 7B/13B/70B model lift
Spearman r = 0.87
```
- Proxy training at ~5% of full compute achieves equivalent prediction quality
- **Cost**: Orders of magnitude cheaper than full training

#### AUM (Area Under the Margin)
```
AUM = mean(gap between correct class confidence and next-highest alternative, over all epochs)
Negative AUM = mislabeled/unanswerable
RL analog: max_rollout_reward − mean_rollout_reward
```

#### Forgetting Events
```
Counts transitions from correct → incorrect between training steps
Unforgettable examples (zero forgetting) can be pruned with <0.5% loss
RL analog: pass rate oscillation across rollout batches
```

#### Dataset Cartography
```
Maps tasks onto (confidence, variability) axes from training checkpoints
"Ambiguous" zone (high variability, intermediate confidence) = 90% performance at ~30% data
Justifies RL pass rate band (0.2, 0.8)
```

---

## NOVEL SIGNALS (Proposed, Computable with ≤N Rollouts)

### 8.1 Solution-Path Diversity Score (SPDS)
```
SPDS(T) = 1 - mean_cosine_similarity(chain_i, chain_j | r_i = r_j = 1)
```
- Collects N rollouts, filters to successful ones, embeds reasoning chains
- High SPDS → multiple distinct solutions → richer gradient landscape
- **Cost**: N rollouts (already computed) + sentence encoding of successful chains
- **Limitation**: Requires ≥2 successful rollouts; filter to p(T) > 0.15

### 8.2 Coherence-Without-Correctness Score (CWC)
```
CWC(T) = mean(1/perplexity_M(y_wrong) for y_wrong in failed_rollouts)
CWC_norm(T) = CWC(T) / baseline_perplexity
```
- High perplexity on failed outputs = incoherent noise (OOD)
- Low perplexity on failed outputs = coherent misconception (learnable)
- **Cost**: One forward pass per failed rollout; <10% overhead
- **Normalized**: CWC_norm > 0.5 = high learning potential

### 8.3 Embedding Eccentricity Score (EES)
```
EES(cohort) = 1 - cosine_similarity(centroid(embed(cohort)), reference_centroid)
```
- Measures distance of cohort from training distribution
- Optimal range: EES ∈ (0.15, 0.60)
  - Below 0.15: redundant with training corpus
  - Above 0.60: likely out-of-distribution
- **Cost**: O(n) embedding passes; pre-computed reference centroid adds zero cost

### 8.4 Delta-Loss on 1% Probe (DLP)
```
DLP(cohort) = loss(M_before, val) - loss(M_after_1step, val)
where 1 gradient step applied to 1% of cohort (min 5 tasks, last-layer only)
```
- Positive DLP = gradient directions aligned with validation objective
- DLP ≈ 0 = redundant gradient directions
- DLP < 0 = adversarial/OOD
- **Cost**: ~2 min CPU for small models; seconds on GPU (requires checkpoint management)
- **Use case**: Second-pass filter after pass rate and diversity screening

### 8.5 Difficulty Dispersion Score (DDS)
```
DDS(cohort) = Var({ p(T) : T ∈ cohort })
Report with mean(p) together.
```
- Cohort with all p ≈ 0.5 → uniform difficulty → saturates narrow band
- High DDS → tasks span full difficulty range → natural curriculum
- Ideal: High DDS + centered mean pass rate
- **Cost**: Zero additional cost (free second-order statistic from pass rates)
- **Guard**: Reject if DDS > 0.3 AND (mean_p < 0.1 OR mean_p > 0.9) [bimodal-trivial pattern]

### 8.6 Cross-Rollout Reasoning Coherence (CRC)
```
CRC(T) = mean_cosine_similarity(chain_i, chain_j for all i,j pairs)
```
- Unlike SPDS (successful only), CRC measures all rollout diversity
- Low CRC = model genuinely explores solution space → richer gradient landscape
- High CRC = model always follows same path → locked representation
- **Cost**: N rollouts + sentence embeddings for all chains + N² cosines (negligible)
- **Limitation**: Extract post-prompt reasoning to avoid artificial similarity from repeated prompt text

---

## DATASET-LEVEL METRICS

### Learning Zone Metrics
```
Learning zone fraction = fraction of tasks with 0.15 < p(T) < 0.85
Target: > 0.5

Hard filter = fraction with p(T) < 0.05 or p(T) > 0.95
Target: < 0.2 total

Intermediate concentration = fraction at 0.3 < p < 0.7
Higher = more efficient training
```

### Optimal Pass Rate Histogram
Not uniform. Empirically optimal distribution [arxiv:2605.05112]:
- **30% easy** (high p)
- **50% medium** (0.3 < p < 0.7)
- **20% hard** (low p)
This distribution outperforms uniform in convergence speed and final accuracy.

### Redundancy Metrics
- **SimHash**: Constant-time approximate cosine via LSH
- **MinHash**: Jaccard similarity over token-sets; threshold J(Tᵢ, Tⱼ) > 0.85 for near-dup
- **Embedding cosine dedup**: Catches semantic near-duplicates (same reasoning, different surface form)
- **Action**: If near-dup fraction > 30%, deduplicate before training

---

## KEY EMPIRICAL RESULTS & NUMBERS

### Pass Rate Theory & Practice
- **Optimal single-task p**: 0.5 (maximum p(1-p) = 0.25)
- **Optimal cohort range**: 0.3 < mean(p) < 0.7
- **+12% math accuracy** with difficulty filtering 0.1 < p < 0.9 reaching target in <50% steps [arxiv:2504.03380]
- **2x speedup on SWE-bench** with pass-rate control (30% easy + 50% medium + 20% hard) [arxiv:2605.05112]
- **<5% gradient contribution** from tasks outside (0.1, 0.9) [arxiv:2605.05112]

### Gradient Energy Bound (Theoretical)
```
E[||∇θ L_GRPO||²] ≥ λ · E_{T∈C}[p(T) · (1 - p(T))]
```
Cohorts maximizing E_T[p(T)(1-p(T))] converge faster per training step.

### Proxy Model Predictivity
- **Spearman correlation (1.3B proxy → 7B/13B/70B)**: r = 0.87
- **Proxy training cost**: ~5% of full RL training compute
- **Enables oracle prediction** without full training run cost

### Early Convergence Forecasting
- **Pearson correlation** (accuracy at 5% training → final accuracy): r = 0.91 [arxiv:2605.18607]
- **Enables early stopping** to cancel unpromising cohorts at minimal cost

### Data Attribution Methods
- **EL2N**: Pruning 30–50% of ImageNet with EL2N matches full-dataset performance; R² > 0.9 for early-epoch prediction
- **DataInf**: Pearson r = 0.97 with ground-truth influence; ~100x faster than full Hessian-based influence
- **TRAK**: Near-full-influence accuracy at 100x lower cost than traditional methods
- **DavIR**: 6% of Alpaca (3K/52K) selected by RHO-Loss criteria matches full 52K on instruction-following

### Diversity & Deduplication
- **Vendi Score quality threshold**: VS/n > 0.3 (lower ratios → entropy collapse)
- **Task Diversity Coefficient quality threshold**: DC > 0.4 (mean pairwise embedding distance)
- **MinHash near-dup detection**: ~100% precision at O(n log n) cost

### Dataset Cartography (SFT Analog)
- **"Ambiguous" zone** (high variability, intermediate confidence): **90% of full-data performance** at ~30% of data
- Justifies RL pass-rate band of (0.2, 0.8)

---

## COMPOSITE VALUE INDEX (CVI) WEIGHTS

```
CVI = 0.30 × pass_rate_learning_zone_fraction
    + 0.20 × normalized_reward_variance
    + 0.15 × normalized_difficulty_dispersion
    + 0.20 × vendi_diversity_normalized
    + 0.15 × coherence_without_correctness_normalized
```

### Normalization Schemes
- **Learning Zone Fraction**: direct [0,1], fraction of tasks with 0.15 < p < 0.85
- **Reward Variance**: min(mean_var × 4, 1.0) [scale assumes var ∈ [0, 0.25]]
- **DDS**: min(dds × 4, 1.0) [scale assumes dds ∈ [0, 0.25]]
- **Vendi**: min(vendi / n_tasks, 1.0)
- **CWC**: min(mean_cwc × 10, 1.0) [scale assumes cwc ∈ [0, 0.1]]

### Hard Veto Conditions (Reject Before Computing CVI)
1. All trivial: mean p > 0.95 across all tasks → zero gradient
2. All unsolvable: max score = 0 for all tasks → no learning signal
3. High near-dup: MinHash dup fraction > 50% → deduplicate first
4. OOD cohort: EES > 0.70 → flag for manual review

### CVI Calibration Protocol
After each RL training run:
1. Record: (CVI_score, actual_lift)
2. After ≥5 runs, fit isotonic regression: lift_predicted = isotonic_regressor(CVI)
3. Identifies domain-specific adjustments to signal weights

### Computational Cost (500-task cohort, N=20 rollouts)
- Rollout computation: ~10,000 forward passes → 5–15 min on A100
- Embedding computation (Vendi, EES): ~500 encoder passes → 30 sec CPU
- CVI arithmetic: <1 sec
- **Total: 5–15 minutes** (vs. hours for actual RL training)

---

## HARDENING AGAINST FAILURE MODES

### Entropy Collapse (Primary RL Failure Mode)
- **Mechanism**: Low-diversity cohorts → policy collapses to narrow strategy repertoire
- **Prevention**: Require Vendi Score VS/n > 0.3; Task Diversity Coefficient DC > 0.4
- **Proxy indicator**: High-value tasks should show CRC < 0.5 (model explores multiple paths)

### Bimodal-Trivial Distribution
- **Problem**: DDS high but mean_p at extremes (30% trivial + 30% unsolvable + 40% elsewhere)
- **Guard**: Reject if DDS > 0.3 AND (mean_p < 0.1 OR mean_p > 0.9)
- **Fix**: Re-sample to centered distribution around p = 0.4–0.6

### Out-of-Distribution Cohort
- **Signal**: EES > 0.60 indicates distribution too far from training manifold
- **Action**: Flag for manual review; consider mixing with in-distribution examples

### Learnable vs. Noise Failures
- **CWC Score** distinguishes:
  - Low perplexity failures = coherent misconceptions (learnable, CWC_norm > 0.5)
  - High perplexity failures = OOD noise (not learnable, CWC_norm < 0.3)
- **Use**: Gate on CWC_norm in cohort filter

---

## COST COMPARISON FOR PRACTICAL SELECTION

| Approach | Cost | Predictive Power | Throughput |
|----------|------|-----------------|-----------|
| Pass rate (N=20) | ~10 min (500 tasks) | High | ~50 tasks/min |
| CVI full pipeline | ~15 min (500 tasks) | Very High | ~33 tasks/min |
| One backward pass (TRAK/gradient alignment) | ~2 hours (500 tasks) | High | ~4 tasks/min |
| Proxy RL lift (1.3B model) | ~1 GPU-day | Extremely High (r=0.87) | 1 task/GPU-day |
| Full RL training (oracle) | ~days | Perfect | 1 task/days |

**Recommendation**: Use CVI (pass rate + Vendi + DDS + CWC) as primary gate. Use proxy RL lift for final validation before committing to full training run.

---

## REFERENCES TO GROUND PRESENTATION

**Key Empirical Papers**:
- [arxiv:2504.03380] Online Difficulty Filtering for Reasoning-Oriented RL (EACL 2026) — +12% math reasoning
- [arxiv:2605.05112] Rollout Pass-Rate Control for Efficient RL — 2x speedup on SWE-bench
- [arxiv:2509.21013] Proxy Model Prediction of RL Reasoning Performance — r=0.87 Spearman
- [arxiv:2605.18607] Forecasting Downstream Performance Before Full Training — r=0.91 Pearson

**Foundational Work** (cited in original paper):
- EL2N [Paul et al., NeurIPS 2021, arxiv:2107.07075]
- AUM [Pleiss et al., NeurIPS 2020, arxiv:2001.10528]
- Forgetting Events [Toneva et al., ICLR 2019, arxiv:1812.05159]
- Dataset Cartography [Swayamditta et al., EMNLP 2020, arxiv:2009.10795]
- TRAK [Park et al., ICML 2023, arxiv:2303.14186]
- DataInf [Kwon et al., ICLR 2024, arxiv:2310.00902]
- RHO-Loss [Mindermann et al., ICML 2022, arxiv:2206.07137]
- Vendi Score [Friedman & Dieng, TMLR 2023, arxiv:2210.02410]

---

## PRESENTATION ANGLES

### For Practitioners
Lead with **CVI framework**: 5 weighted signals, 15 min computation, direct lift prediction. Provides actionable gate without full training.

### For Researchers
Emphasize **gradient energy bound** (p(1-p) theory) + **proxy model prediction** (r=0.87). Novel signals (SPDS, CWC, EES, DDS, CRC) provide new failure-mode detection.

### For ML Systems Design
Highlight **cost-quality Pareto frontier**: pass rate achieves "break" (high power, low cost). Online pass-rate control → 2x speedup. Synergizes with GRPO/curriculum learning.

### For Hackathon Demo
Implement: pass rate filter → Vendi Score → CVI computation → top-N cohort selection. Show CVI predictions vs. held-out ground truth if available.
