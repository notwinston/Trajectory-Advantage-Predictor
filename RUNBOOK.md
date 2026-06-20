# TAP v1 — Wave 2 Runbook (collection launch + handoff)

This runbook hands off the hardened Prime Intellect launcher for the TAP v1
data collection. It covers the **one-time unblock**, the **1xH100 smoke**, the
**exact full-run command**, and how to monitor / reap / continue to Wave 3.

---

## ⛔ BLOCKED-ON-USER: `PRIME_API_KEY` is not in this environment

The 1xH100 smoke could **not** run because Prime Intellect is not authenticated
in this container:

```
$ prime availability list --gpu-type H100_80GB --output json
Error fetching availability page 1: No API key configured. Use command 'prime login' ...
{ "gpu_resources": [], "total_count": 0, ... }   # empty -> auth failure, not a pass
```

`PRIME_API_KEY` is absent from the environment, from `~/.prime` / `~/.config/prime`,
and from any `.env`; the Claude Docker wrapper only forwards `CLAUDOCKER_*`/OAuth,
so it never injects a PI key. The CPU integration gate and the launcher hardening
all pass — the **only** thing missing is the credential.

### To unblock (user action, ~1 minute)

Pick ONE:

```bash
# (a) Export the key into the environment (what the launcher/CLI read):
export PRIME_API_KEY=<your-prime-intellect-api-key>

# (b) …or store it in the prime CLI config:
prime config set-api-key        # prompts securely
# prime login                   # interactive browser OAuth (alternative)
```

Then confirm auth + the registered SSH key:

```bash
prime whoami
prime availability list --gpu-type H100_80GB --output json | jq '.gpu_resources | length'   # must be > 0
ssh-keygen -y -f /workspace/private_key.pem        # must match /workspace/public_key.pem
```

The public key `/workspace/public_key.pem` (comment `prime-intellect`) should
already be registered on the PI account. If a provisioned pod is later
unreachable over SSH, register that public key in the PI dashboard
(Settings → SSH keys) and re-run.

After `availability list` returns a non-empty array, run the smoke (below).

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
