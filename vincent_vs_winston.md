# TAP v1 — Branch Comparison: `vincent` vs `winston-math500`

Read-only comparison of two parallel TAP v1 implementations. Findings come from an
independent multi-agent read of **both** trees plus running **both** test suites;
wrong analyst claims were corrected against the code. Citations are `path:line`.

- **`vincent`** (`dbf6617 model`): unified `tap_loop/` package + `run_tap_prime.py`; keeps base `math_loop/`.
- **`winston-math500`** (`e21c858`): `math_loop/` engine + `tap/` eval + `math_loop/tap_controller.py` driver + hardened `run_prime_rl_math_loop.py` + `reap_pods.py`.

Both branched from `f744925`; `vincent` adds one commit, `winston` adds the full Wave 0/1a/1b + driver + launcher stack.

---

## 1. Executive summary

The core architectural split is **who runs the GRPO step**.

- **`vincent`** is a unified single-process engine that **re-implements GRPO in Python**
  (`tap_loop/fixed_update.py:43-133`), samples rollouts in-process via `model.generate()`
  (`tap_loop/collector.py:221`), and writes Parquet directly — elegant, self-contained, easy to read.
- **`winston`** decouples collection from evaluation: it **shells out to real prime-rl** for every
  state and branch using the weights-only recipe (`math_loop/tap_controller.py:421-429`), persists a
  raw-artifact tree, and a separate feature extractor (`math_loop/features.py`) converts it to a
  frozen, schema-validated Parquet contract.

**Bottom line:** `winston` is the stronger *production* implementation — real labels/probes (real
LoRA-gradient sketch, real fingerprint, real Adam moments, real generic-KL), a fail-closed
cost/teardown harness, a frozen schema with strict validation, a committed fixture, and 79 passing
tests. `vincent` is the cleaner *codebase* but its real-pod path is undermined by a decisive
correctness problem — **several TAP features are synthesized random noise even in the torch backend**
— and its launcher can leak expensive pods on failure. Recommendation: keep `winston` as the base,
selectively borrow `vincent`'s clarity (see §6).

---

## 2. Dimension-by-dimension verdict

| Dimension | Stronger | Why |
|---|---|---|
| Collection driver | ~ `winston` | Real prime-rl GRPO (no reimplementation risk, `tap_controller.py:428`); `vincent`'s hand-rolled surrogate (`fixed_update.py:21-40`) is more debuggable but a *second* GRPO that can drift. Both loops are fully **sequential** (no parallelism in either). |
| Labels / probes | `winston` (clearly) | Real gradient sketch (`branch.py:248-268`), real fingerprint (`tap_probes.py:208-231`), real Adam moments (`tap_controller.py:278-300`), real generic-KL (`tap_probes.py:166-205`) vs `vincent`'s stubs. Utility formula **identical** + spec-correct in both. |
| Model / training / eval | ~ tie, slight `winston` | Same loss (Huber + 0.5× within-state pairwise rank). `winston` adds NaN-safe empty-history attention masking (`model.py:129-136`) + 5 learned baselines vs `vincent`'s 1. |
| Data / schema / artifacts | `winston` | Frozen schema + strict post-Parquet validator (column order, dtypes, vector widths, NaN guard, `schema.py:217-274`); `vincent` validates dims only at write (`artifacts.py:12-19`). |
| Launcher / ops / safety | `winston` (far) | Fail-closed cost+wall-clock monitor, SIGTERM/atexit teardown, required commit pin, `TAP_ALLOW_FULL_RUN` gate, prefix reaping. `vincent` has **none** and defaults to datacrunch (`run_tap_prime.py:23`). |
| Tests / validation | `winston` (far) | Verified counts: **`vincent` 17, `winston` 79**. `winston` ships a committed `raw_artifacts` fixture; `vincent` has one dry-run e2e test and no feature extractor. |

---

## 3. Architecture detail

### `vincent` — in-process torch engine (`tap_loop/`)
- `PolicyBackend` Protocol with `DryRunPolicyBackend` (CPU) + `TorchPolicyBackend` (real, GPU) — `collector.py:41,65,171`.
- `TorchPolicyBackend.generate_candidate_trajectories` (`collector.py:207-300`): `model.generate(do_sample=True, num_return_sequences=group_size)`, exact-match reward, within-group z-scored advantage, and **real per-token logprobs/entropy/hidden-state embeddings** from the forward pass (`:251-257`).
- `apply_fixed_rollout_lora_update` (`fixed_update.py:43-133`): loads the before-adapter via `PeftModel(is_trainable=True)`, one `AdamW` step on the clipped GRPO surrogate, saves adapter + `optimizer.pt`.
- Default backend is **`torch`**, not dry-run (`run_tap_prime.py:443`) — real runs use the torch path.

### `winston` — prime-rl driver + raw-tree contract (`math_loop/` + `tap/`)
- `run_controller` (`tap_controller.py`): generate a state via prime-rl, branch each candidate **weights-only** (`model_name`=state ckpt, `max_steps=1`, `renderer="default"`, run_default layout — Mark's `fresh-branch-weights` recipe), read prime-rl's persisted `train_rollouts.jsonl`, score probes, write the raw tree.
- `math_loop/features.py` converts the raw tree → 4 Parquet against the frozen `tap/schema.py` contract.
- `tap/` = SmallTAP (attention MLP) + ridge/GBT/no-history/numeric/candidate baselines + within-state ranking eval + `run_all` (CSV + report).

---

## 4. What `vincent` does better

- **Cleaner, unified mental model** — collection, GRPO, probes, training, metrics all in one `tap_loop/` package.
- **Real per-token trajectory features** — real logprobs/entropy/embeddings (`collector.py:251-257`). `winston` **falls back to 0.0** here because prime-rl hardcodes per-token logprobs to 0.0.
- **Explicit optimizer control** — saves/loads `optimizer.pt`, so Adam moments resume (`fixed_update.py:84,128`); `winston` branches weights-only.
- **Stricter data guards** — rejects out-of-range MATH levels (`data.py:74-75`) + explicit `assert_no_math500_leakage` (`tests/test_tap_loop.py:10`).
- **Fully CPU-testable real loop** via `DryRunPolicyBackend`, and a cleaner **state-manifest resume** (`collector.py:546-566`).
- **No external prime-rl version coupling** for branching.

## 5. What `winston` does better

- **Real candidate labels** (gradient sketch, fingerprint, Adam moments, generic-KL) vs `vincent`'s noise/zeros.
- **Real prime-rl GRPO** for state-gen and branches (no surrogate reimplementation).
- **Fail-closed cost/teardown harness** (`run_prime_rl_math_loop.py:335-393`) + required commit pin + signal handlers + pod reaping (`reap_pods.py`).
- **Frozen schema + strict post-Parquet validator** with NaN guard (`schema.py:217-274`).
- **Committed fixture + 79 tests** exercising raw→Parquet bridge, schema, baselines, model convergence, and launcher safety rails.
- **NaN-safe model** (empty-history attention masking) + multi-layer NaN/inf defenses in feature extraction.

---

## 6. Correctness bugs / spec deviations

### `vincent` (significant)
1. **Synthetic features in the *real* torch path (decisive).** `_candidate_rows` hardcodes
   `gradient_sketch = rng.uniform(-0.5,0.5)` (`collector.py:448`),
   `matched_probe_gradient_alignment = rng.uniform(-1,1)` (`:488`),
   `candidate_log_probability_change = rng.uniform(-0.2,0.2)` (`:489`); dependent features
   `gradient_norm`, `max/mean_gradient_similarity_to_history` (`:479-484`) are therefore noise.
   `policy_fingerprint=[0.0]*16` and `adam_*_moment_norm=0.0` (`:599-601`). These feed both the model
   and the `gradient_norm`/`gradient_alignment` baselines, so any skill there is **spurious**. The
   in-process backward pass *could* produce a real sketch (the plumbing exists) but isn't wired in.
2. **`generic_kl` is an NLL, not a KL.** `evaluate_before_state` returns `generic.nll` as `generic_kl`
   (`collector.py:325`), then `utility_points` treats it as the KL penalty (`probes.py:97-99`). A
   correct `average_token_kl` exists (`probes.py:109-121`) but is never called. `winston` computes a
   true KL(base‖branch) (`tap_probes.py:166-205`).
3. **Pod leak on failure.** `if args.keep_pod or not mirror_ok:` keeps the pod when the run *fails*
   (`run_tap_prime.py:489`); no signal handler, no atexit, no cost monitor → a crash or Ctrl+C strands
   a 4×H100. Biggest operational risk in either tree.
4. **Cross-branch incompatibility:** `trajectory_embedding` dim is 256 in `vincent`, 128 in `winston`.

### `winston` (minor)
1. **Silent zero-fill** on any gradient/optimizer error (`branch.py:269`, `tap_controller.py:298`) —
   safe for the pipeline but hides CUDA OOM / load failures.
2. **`read_rollouts` untested against real prime-rl** — reward/advantage pulled via key-fallback
   (`tap_controller.py:249`); if prime-rl renames its rollout fields, rewards silently become 0.0.
   This is the load-bearing seam the smoke would have validated.
3. **Schema validator checks structure, not referential integrity** (`schema.py:217-274`).
4. **`generic_before = 0.0` hardcoded** (`tap_controller.py:398`) — correct given `generic_incremental_kl`
   already returns the increment, but an implicit coupling worth a comment.

---

## 7. Recommendation

**Keep `winston` as the base and borrow two things from `vincent`:**
- Its **state-manifest resume** design (`collector.py:546-566`) — cleaner than `winston`'s per-candidate
  `probe_after.json` existence check.
- Its explicit **`assert_no_math500_leakage`** runtime guard — add an equivalent to `winston`'s data prep.

**Honest caveat the other way:** `winston`'s real-prime-rl rollout-read path is the one thing not yet
validated on a pod (the smoke was stopped), whereas `vincent`'s torch path at least runs end-to-end —
albeit with several stubbed features.

**Attractive hybrid:** `vincent`'s in-process backend already has the backward pass, so it could compute
the **real** gradient sketch and feed `winston`'s frozen schema + hardened launcher + eval — that would
beat both.

### Top 3 to verify before trusting `winston` on a real pod
1. **prime-rl rollout schema:** one real branch confirms `read_rollouts`/`_map_rollout_row`
   (`tap_controller.py:221-275`) pull **non-zero** reward + advantage from `train_rollouts.jsonl` at the pin.
2. **GPU label sanity:** `compute_lora_gradient_sketch`, `compute_policy_fingerprint`,
   `_optimizer_moment_norms`, `generic_incremental_kl` return **non-degenerate** values (not the silent
   zero-fallbacks) — else the gradient/fingerprint features quietly become zero like `vincent`'s.
3. **Cost monitor pricing:** the chosen offer's `price_per_hour` is populated so `CostWallclockMonitor`
   enforces the cap rather than fail-closing at startup (`run_prime_rl_math_loop.py:366-371`).

(If shipping `vincent` instead, the #1 fix is replacing the synthetic features at `collector.py:448,488,489`
and the keep-pod-on-failure leak at `run_tap_prime.py:489` — both are blockers.)
