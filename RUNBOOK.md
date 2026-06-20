# TAP v1 — Collection Runbook (driver + launcher + pod handoff)

The TAP v1 data-collection driver is **implemented and CPU-validated end to end**,
and the Prime Intellect launcher is hardened and pod-validated through bootstrap +
pre-flight on a real 2×H100. This runbook is the exact path to collect labels and
run Wave 3.

## Status

- **Collection driver** (`math_loop/tap_controller.py::run_controller`) — implemented
  on the proven *fresh-branch-weights* recipe: generate a state, branch each
  candidate weights-only (`model_name` = the state checkpoint, one fresh GRPO step,
  `run_default/` checkpoint layout — no optimizer resume), read prime-rl's persisted
  rollouts, score matched/global probes + generic-KL before/after, compute the policy
  fingerprint and LoRA gradient sketch, and write the raw-artifact tree
  `math_loop.features` converts to the four Parquet files.
- **CPU contract — VALIDATED**: a synthetic raw tree at the smoke shape (2 chains ×
  1 state × 2 candidates) passes `features.convert` → `tap.schema --validate` →
  `tap.run_all` (exit 0, real TAP-vs-baseline metrics). See
  `python -m unittest tests.test_tap_schema tests.test_tap_engine tests.test_tap`.
- **Pod path — VALIDATED through bootstrap + pre-flight** on a 2×H100 lambdalabs pod
  (`prime whoami` → winston thov; availability returns offers): apt-under-lock,
  prime-rl pinned to `4d361adacc5d4984ffe2c912cc987690ed1aeff7` with submodules
  (incl. `verifiers`), `uv sync --all-extras`, `peft` installed, and the
  not-degradable import (`verifiers` + `prime_rl.orchestrator.advantage.AdvantageOutputs`)
  resolves. The full GPU collection itself was **not run to completion** (stopped to
  prepare the repo for push); run the smoke below to execute it.
- **Credential**: `PRIME_API_KEY` must be in the environment (`export PRIME_API_KEY=…`).
  Never write it to a tracked file.

## Pod-validated facts the launcher now bakes in

- prime-rl needs **2 GPUs** (1 trainer + 1 inference; `gpus_per_node=2`), so the
  smoke runs on **2×H100**, not 1.
- lambdalabs boots as the `ubuntu` sudo user with **no `/workspace`** — the bootstrap
  now creates it (sudo) and uses `sudo` for apt when not root.
- prime-rl submodules use `git@github.com:` URLs; the bootstrap rewrites them to
  https (`GIT_CONFIG insteadOf`) so they clone without an SSH key.
- `peft` is **not** a prime-rl dependency but the TAP probes need it (to load
  prime-rl's separately-saved LoRA adapter); the bootstrap now `uv pip install peft`.
- `[wandb]` is removed from the config and `WANDB_MODE=disabled` is exported, so a
  fresh pod doesn't block on a wandb login.

## Step 1 — resolve + pin the prime-rl commit

```bash
PRIME_RL_COMMIT=$(git ls-remote https://github.com/PrimeIntellect-ai/prime-rl HEAD | cut -f1)
# validated commit: 4d361adacc5d4984ffe2c912cc987690ed1aeff7
```

## Step 2 — 1× smoke (2×H100, 2 chains × 1 state × 2 candidates, ≤ cap, auto-reaped)

```bash
export PRIME_API_KEY=<key>
python run_prime_rl_math_loop.py \
  --smoke \
  --provider lambdalabs --gpu-type H100_80GB \
  --prime-rl-commit "$PRIME_RL_COMMIT" \
  --max-cost-usd 40 --max-wallclock-seconds 14400 \
  --ssh-key /workspace/private_key.pem
```

The launcher provisions, bootstraps (pinned prime-rl + submodules + peft), runs the
pre-flight import, drives the collection + featurization, downloads, and validates
the mini-Parquet with `tap.schema --validate` + `tap.run_all` — tearing the pod down
on success, failure, signal, or cost/wall-clock breach. **Success** = `tap.schema`
prints `OK`, `tap.run_all` prints a verdict, and `python reap_pods.py --list` shows
`ACTIVE tap-v1- pods: 0`.

## Step 3 — full collection (USER-triggered; 192 labels)

The loop never sets `TAP_ALLOW_FULL_RUN`, so only a human can launch this.

```
TAP_ALLOW_FULL_RUN=1 python run_prime_rl_math_loop.py \
  --provider lambdalabs --gpu-type H100_80GB --gpu-count 2 \
  --prime-rl-commit <sha> \
  --max-cost-usd 80 \
  --states 8 --chains 3 --candidates-per-state 8 \
  --ssh-key /workspace/private_key.pem \
  --output-dir outputs --run-id <run_id>
```

3 chains × 8 states × 8 candidates = **192 labels** (the TAP v1 design scale;
you can still override the shape for a shorter run). The driver branches serially; scaling branches across
more GPUs is a future optimization. Collected Parquet lands at:

```
PARQUET_DIR=outputs/tap/<run_id>/parquet
```

## Step 4 — train + evaluate TAP on the real collection (Wave 3)

```bash
VIRTUAL_ENV=$PWD/.venv ~/.local/bin/uv run --no-project \
  python -m tap.schema --validate "$PARQUET_DIR"
VIRTUAL_ENV=$PWD/.venv ~/.local/bin/uv run --no-project \
  python -m tap.run_all --parquet-dir "$PARQUET_DIR" --out outputs/tap_report_real
```

`tap.run_all` does the chain-0↔chain-1 cross-split, trains SmallTAP + the ridge /
GBT / no-history / numeric-only / candidate-only baselines and the heuristic
selectors, and writes `results.csv` + `report.{md,tex,pdf}`. The headline comparison
is TAP vs reward-only vs probability-only vs random.

## Monitoring / reaping

```bash
tail -f outputs/<...>/pod.log          # streamed bootstrap + collection logs
python reap_pods.py --list             # active tap-v1- pods + count
python reap_pods.py --terminate-prefix tap-v1-smoke-   # reap all smoke pods
python reap_pods.py --terminate-id <pod_id>            # reap one pod by id
```

`--terminate-prefix` refuses any empty or non-`tap-v1-` prefix. The launcher's
fail-closed monitor reaps the pod and exits non-zero on cost-cap, wall-clock, or
monitor-failure, appending a banner here.

## Safety summary (enforced by the launcher)

- Default provider `lambdalabs`; prime-rl pinned via the required `--prime-rl-commit`.
- `--max-cost-usd` ≤ $80, enforced by a fail-closed cost monitor **and** an
  independent wall-clock deadline.
- `pod_id.txt` on create; atexit + SIGINT/SIGTERM + monitor breach reap by
  `tap-v1-smoke-` prefix **and** id.
- `--keep-pod` forbidden; any non-`--smoke` run requires `TAP_ALLOW_FULL_RUN=1`.
- SSH via the provided `/workspace/private_key.pem` (chmod 600, verified against
  `/workspace/public_key.pem`).
- rsync-free `tar`-over-ssh upload/download fallback when rsync is absent locally.
