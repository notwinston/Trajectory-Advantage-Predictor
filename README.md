# Inference_Time

## Basic prime-rl Qwen3 MATH loop

This repo includes a minimal prime-rl setup for a Qwen3-8B LoRA/GRPO MATH
branch-labeling loop:

```bash
python run_prime_rl_math_loop.py --gpu-type H100_80GB --gpu-count 2 --output-dir outputs/qwen3_math_loop
```

The local controller can also be run directly inside a prime-rl checkout:

```bash
PYTHONPATH=$PWD uv run python -m math_loop.controller \
  --config configs/prime_rl/qwen3_8b_math_state.toml \
  --states 48 \
  --candidates-per-state 16 \
  --batch-prompts 4 \
  --group-size 4
```

The controller prepares MATH train/probe splits only. MATH-500 is intentionally
kept out of the training/label loop and is prepared or used only for final eval:

```bash
python -m math_loop.data --include-final
python -m math_loop.final_eval --checkpoint outputs/qwen3_math_loop/states/weights/step_48 --split data/math_loop/math500.jsonl
```

Useful local checks:

```bash
python run_prime_rl_math_loop.py --dry-run
python -m unittest discover
python -m py_compile run_prime_rl_math_loop.py math_loop/*.py tests/*.py
```

## TAP v1 Prime Intellect pipeline

This repo also includes a TAP v1 fixed-rollout pipeline under `tap_loop/`.
It is separate from the original branch-labeling loop because TAP candidates
must use the exact generated trajectories for both feature extraction and the
one-step branch update.

### What the TAP run does

The default run collects 72 TAP labels:

- 2 chains
- 6 policy states per chain
- 6 candidate updates per state
- 2 MATH prompts per candidate
- 4 completions per prompt
- Qwen3-8B LoRA rank 16, BF16, non-thinking mode, 192 completion tokens

For each state, the collector writes the before-state row, generates fixed
candidate trajectories, applies one fixed-rollout branch update per candidate,
labels the branch with probe utility, randomly promotes one selected branch as
the next chain state, then compacts artifacts. After collection,
`tap_loop.train_tap` trains/evaluates the baseline rankers and the small TAP
model when `torch` is available, writing `reports/tap_metrics.json`.

### Prerequisites

Install and authenticate the Prime CLI locally:

```bash
uv tool install prime
prime login
prime config set-ssh-key-path ~/.ssh/id_rsa
```

Export a Hugging Face token that can read `Qwen/Qwen3-8B`:

```bash
export HF_TOKEN=<your-hugging-face-token>
```

Create or choose a Prime persistent disk in the same provider/datacenter as the
H100 offer. Persistent disk is the recommended default because TAP checkpoints,
Parquet fragments, logs, and HF cache can survive pod termination.

### Local dry-run

This checks command construction and preflight behavior without creating a pod:

```bash
python3 run_tap_prime.py \
  --dry-run \
  --ephemeral-ok \
  --backend dry-run \
  --run-id tap_smoke \
  --chains 1 \
  --states-per-chain 1 \
  --candidates-per-state 2
```

### Remote smoke test

Run this before spending the full H100 budget. It uses the real remote setup but
only collects 2 candidate labels:

```bash
RUN_ID=tap_smoke_$(date -u +%Y%m%d_%H%M%S)

python3 run_tap_prime.py \
  --gpu-count 4 \
  --disk-id <prime-disk-id> \
  --run-id "$RUN_ID" \
  --chains 1 \
  --states-per-chain 1 \
  --candidates-per-state 2 \
  --batch-prompts 2 \
  --group-size 4 \
  --max-completion-tokens 192 \
  --sync-interval-min 5 \
  --keep-pod \
  --output-dir "outputs/tap_v1/$RUN_ID"
```

Inspect:

```bash
cat "outputs/tap_v1/$RUN_ID/collection_status.json"
cat "outputs/tap_v1/$RUN_ID/reports/tap_metrics.json"
find "outputs/tap_v1/$RUN_ID/checkpoints/chains" -name state.json -print
```

If the smoke test succeeds and the mirror is complete, terminate the kept pod
from Prime. Keeping the pod for the first smoke run makes debugging much easier
if dependency installation, Hugging Face access, or mount paths are wrong.

### Full TAP training run

```bash
RUN_ID=tap_v1_$(date -u +%Y%m%d_%H%M%S)

python3 run_tap_prime.py \
  --gpu-count 4 \
  --disk-id <prime-disk-id> \
  --run-id "$RUN_ID" \
  --states-per-chain 6 \
  --candidates-per-state 6 \
  --batch-prompts 2 \
  --group-size 4 \
  --max-completion-tokens 192 \
  --sync-interval-min 10 \
  --output-dir "outputs/tap_v1/$RUN_ID"
```

Use 3 H100s if 4 are unavailable:

```bash
python3 run_tap_prime.py --gpu-count 3 --disk-id <prime-disk-id> --run-id "$RUN_ID"
```

The launcher uploads this repo to `/workspace/tap_loop_repo`, clones Prime-RL
under `/workspace/prime-rl`, uses `/mnt/prime_tap/tap_runs/<run_id>` by default
when a persistent disk is attached, mirrors artifacts back to
`outputs/tap_v1/<run_id>`, and keeps the pod alive automatically if artifact
mirroring fails.

### Checkpoints and failure recovery

The TAP pipeline is designed to be rerunnable with the same `RUN_ID`:

- Candidate/state/history rows are written as atomic JSONL fragments under
  `fragments/`.
- Final tables are compacted after every completed state under `parquet/`.
- Branch checkpoints are kept under `checkpoints/branches/<candidate_id>/`.
- After each completed state, the promoted chain state is recorded under
  `checkpoints/chains/chain_XX/state_YYY/state.json`.
- `collection_status.json` is updated as a heartbeat while the collector runs.
- The launcher rsyncs artifacts periodically according to `--sync-interval-min`
  and always attempts a final mirror in a `finally` block.

If the pod dies or the job fails midway, rerun with the same `--run-id`,
`--disk-id`, and `--output-dir`:

```bash
python3 run_tap_prime.py \
  --gpu-count 4 \
  --disk-id <prime-disk-id> \
  --run-id "$RUN_ID" \
  --states-per-chain 6 \
  --candidates-per-state 6 \
  --batch-prompts 2 \
  --group-size 4 \
  --max-completion-tokens 192 \
  --sync-interval-min 10 \
  --output-dir "outputs/tap_v1/$RUN_ID"
```

Completed states are skipped using the chain `state.json` manifests. If a
failure happens in the middle of a state, that unfinished state is recomputed,
while earlier completed states are reused.

Avoid deleting these paths until you have copied the final report:

```text
outputs/tap_v1/<run_id>/fragments/
outputs/tap_v1/<run_id>/parquet/
outputs/tap_v1/<run_id>/checkpoints/
outputs/tap_v1/<run_id>/reports/tap_metrics.json
```

### Local checks

```bash
python3 -m unittest discover
env PYTHONPYCACHEPREFIX=/tmp/inference_time_pycache \
  python3 -m py_compile run_prime_rl_math_loop.py run_tap_prime.py math_loop/*.py tap_loop/*.py tests/*.py
```
