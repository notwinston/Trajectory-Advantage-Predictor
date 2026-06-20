# TAP v1 — Wave 2 Runbook (collection launch + handoff)

This runbook hands off the hardened Prime Intellect launcher for the TAP v1
data collection. It covers the **one-time unblock**, the **1xH100 smoke**, the
**exact full-run command**, and how to monitor / reap / continue to Wave 3.

---

## ⛔ BLOCKED-ON-USER: the on-pod collection driver (`tap_controller.run_controller`) is a stub

**The credential is resolved.** A `PRIME_API_KEY` was supplied; the CLI now
authenticates (`prime whoami` → *winston thov*) and `prime availability list
--gpu-type H100_80GB --output json` returns a **non-empty** offers array (8 offers;
4 lambdalabs, incl. a 1×H100 at \$3.29/h). The CPU gate, the launcher hardening,
and the prime-rl pin all pass. **Do not** write the key into any tracked file —
it lives only in the environment (`export PRIME_API_KEY=…`).

**What still blocks the smoke from producing a mini-Parquet:** the real (non
`--dry-run`) collection driver is not implemented. `math_loop.tap_controller.run_controller`
raises `SystemExit("tap_controller real run executes on the Wave 2 GPU pod …")`,
and the per-candidate `_branch_worker` it would dispatch expects pre-generated
`task["rollouts"]`/`probe_before` — i.e. the on-pod rollout + per-token
logprob/entropy extraction against the real prime-rl (plan unknowns a/b/c) must be
built before any labels exist. That file is **read-only in this wave** and the work
is substantial (not a minimal fix), so it cannot be completed here. Spending on a
pod now would only re-confirm this known gap, so no pod was provisioned.

### What was verified (so the driver is the only remaining gap)

- **Auth + availability:** PASS (8 offers; 1×H100 lambdalabs \$3.29/h available).
- **prime-rl pin** `4d361adacc5d4984ffe2c912cc987690ed1aeff7` (current HEAD).
- **Plan unknown (d) — NOT feature-degradable — CONFIRMED at the pin (from source):**
  `class AdvantageOutputs` exists in `prime_rl/orchestrator/advantage.py` and
  `verifiers` is a declared dependency. The pre-flight import
  (`import verifiers; from prime_rl.orchestrator.advantage import AdvantageOutputs`)
  will not block once the driver is built.
- **SSH:** `/workspace/private_key.pem` (chmod 600) `ssh-keygen -y` matches
  `/workspace/public_key.pem` (comment `prime-intellect`).

### To unblock (developer action)

1. Implement the real collection loop in `math_loop/tap_controller.run_controller`
   (or a driver that composes `build_plan` + `_branch_worker` + `math_loop.features`)
   so it generates rollouts, computes before/after probes, writes the raw
   `before/ cand_<k>/ state.json` tree, and advances the main chain. Resolve plan
   unknowns (a) per-token logprob/entropy exposure, (b) `--ckpt.resume-step` Adam
   restore, (c) LoRA-grad backward against the pinned prime-rl, using the W1a
   fallbacks (forward-pass logprobs / weights-only / `grad_unavailable.flag`).
2. `export PRIME_API_KEY=<key>` in the run environment (never on disk in the repo).
3. Confirm `prime whoami` and that `prime availability list … | jq '.gpu_resources|length'` > 0.
4. Run the smoke (Step 2). The hardened launcher will then provision, bootstrap the
   pinned prime-rl, run the pre-flight import, collect, featurize, download, and
   validate — tearing the pod down on success, failure, signal, or cost/time breach.

---

## Step 1 — Resolve and pin the prime-rl commit

The launcher **requires** `--prime-rl-commit`; the smoke pins the commit it
validates and the full run reuses the same pin.

```bash
PRIME_RL_COMMIT=$(git ls-remote https://github.com/PrimeIntellect-ai/prime-rl HEAD | cut -f1)
echo "$PRIME_RL_COMMIT"   # record this; reuse it for the full run
```

## Step 2 — 1xH100 smoke (cheap, ≤ $80, auto-reaped)

```bash
python run_prime_rl_math_loop.py \
  --smoke \
  --provider lambdalabs --gpu-type H100_80GB \
  --prime-rl-commit "$PRIME_RL_COMMIT" \
  --ssh-key /workspace/private_key.pem
```

`--smoke` forces 1 chain × 1 state × 2 candidates on a single GPU. The launcher:
runs an on-pod pre-flight (`import verifiers` + `prime_rl.orchestrator.advantage.AdvantageOutputs`),
collects raw artifacts via `math_loop.tap_controller`, converts them with
`math_loop.features`, downloads them, then validates locally with
`tap.schema --validate` and `tap.run_all`. The pod is torn down on success,
failure, SIGINT/SIGTERM, or any cost/wall-clock breach.

**Success looks like:** `tap.schema --validate` prints `OK: … passes the TAP v1
schema contract`, `tap.run_all` prints a verdict line, and
`python reap_pods.py --list` shows `ACTIVE tap-v1- pods: 0`.

> **Pre-run precondition (integration finding).** `math_loop.tap_controller.run_controller`
> is currently a stub for real (non-`--dry-run`) runs — it raises
> `SystemExit("tap_controller real run executes on the Wave 2 GPU pod …")` and the
> per-candidate `_branch_worker` has no driver loop. That W1a file is read-only in
> this wave, so the real GPU collection loop must be enabled there before the smoke
> or full run will emit labels. Until then the launcher will fail fast on-pod and
> the cost monitor / atexit handler will reap the pod (no money wasted). The
> launcher, reaper, cost/wall-clock guards, schema, featurizer, and eval are all
> verified on CPU and synthetic data.

## Step 3 — Full collection run (USER-triggered; 4×H100; 72 labels)

This is the only way to run beyond the smoke. The loop never sets
`TAP_ALLOW_FULL_RUN`, so it can never launch this itself.

```
TAP_ALLOW_FULL_RUN=1 python run_prime_rl_math_loop.py \
  --provider lambdalabs --gpu-type H100_80GB --gpu-count 4 \
  --prime-rl-commit <sha> \
  --max-cost-usd 80 \
  --states 6 --chains 2 --candidates-per-state 6 \
  --ssh-key /workspace/private_key.pem \
  --output-dir outputs --run-id <run_id>
```

`<sha>` = the `$PRIME_RL_COMMIT` the smoke validated. `<run_id>` = a label you
choose (e.g. `tap_72_run1`). 2 chains × 6 states × 6 candidates = **72 labels**
(8 trajectories each = 576 trajectories). Scale knobs from the spec: stop early
at 48 (`--states 4`) or push to 128 (`--states 8 --candidates-per-state 8`).

After it finishes, the collected Parquet is at:

```
PARQUET_DIR=outputs/tap/<run_id>/parquet
```

## Step 4 — Train + evaluate TAP on the real collection (Wave 3)

```bash
VIRTUAL_ENV=$PWD/.venv ~/.local/bin/uv run --no-project \
  python -m tap.schema --validate "$PARQUET_DIR"
VIRTUAL_ENV=$PWD/.venv ~/.local/bin/uv run --no-project \
  python -m tap.run_all --parquet-dir "$PARQUET_DIR" --out outputs/tap_report_real
```

`tap.run_all` does the chain-0↔chain-1 cross-split, trains SmallTAP + the ridge /
GBT / no-history / numeric-only / candidate-only baselines and the heuristic
selectors, and writes `results.csv` + `report.{md,tex,pdf}` into `--out`. The
headline comparison is TAP vs reward-only vs probability-only vs random
(within-state Spearman, pairwise accuracy, top-1 regret, mean true utility).

---

## Monitoring a live run

```bash
tail -f outputs/<...>/pod.log         # streamed bootstrap + collection logs
prime pods list --output json | jq    # pod state
python reap_pods.py --list            # active tap-v1- pods + count
```

The launcher runs a fail-closed monitor thread: estimated spend
(`offer price/h × elapsed`) ≥ `--max-cost-usd`, OR elapsed ≥ the independent
wall-clock deadline, OR any monitor failure → it reaps the pod and exits non-zero
with a `COST-CAP-HIT` / `WALLCLOCK-HIT` / `MONITOR-FAILURE` banner appended below.

## Reaping pods manually

```bash
python reap_pods.py --list                              # list active tap-v1- pods
python reap_pods.py --terminate-prefix tap-v1-smoke-    # reap all smoke pods
python reap_pods.py --terminate-id <pod_id>             # reap one pod by id
```

`--terminate-prefix` refuses any empty or non-`tap-v1-` prefix. `pod_id.txt`
(gitignored) holds the most recent pod id and is removed after teardown.

## Safety summary (enforced by the launcher)

- Default provider `lambdalabs`, prime-rl pinned to `--prime-rl-commit` (required).
- `--max-cost-usd` ≤ $80, enforced by a fail-closed cost monitor **and** an
  independent wall-clock deadline.
- `pod_id.txt` written on create; atexit + SIGINT/SIGTERM + monitor breach all
  reap by `tap-v1-smoke-` prefix **and** by pod id.
- `--keep-pod` forbidden; `--gpu-count > 1` and any non-`--smoke` run require
  `TAP_ALLOW_FULL_RUN=1` (never set by the loop).
- SSH uses the provided `/workspace/private_key.pem` (chmod 600, verified against
  `/workspace/public_key.pem`) — never a generated key.
