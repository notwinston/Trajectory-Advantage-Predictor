# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Two related subsystems share one codebase:

1. **TAP v1 — Trajectory Advantage Predictor** (the current focus). TAP predicts a
   single scalar, `utility_points`, for a candidate GRPO update applied to a
   Qwen3-8B LoRA policy: how much that update will improve held-out MATH while
   lightly penalizing policy drift. The pipeline collects labeled candidate
   branches on GPU, featurizes them into a frozen 4-Parquet schema, then trains and
   evaluates a small model (`SmallTAP`) against heuristic + learned baselines on
   CPU. The headline result is *TAP vs reward-only vs probability-only vs random*.
2. **The basic prime-rl Qwen3 MATH loop** (`math_loop/controller.py`) — the
   standalone 48-state × 16-candidate branch-labeling loop TAP grew out of. MATH-500
   is deliberately kept out of the train/label loop and used only for final eval
   (`math_loop/final_eval.py`).

`docs/Implementation_Report/report.md` is the authoritative architecture spec; read
it before changing collection, schema, or eval. `RUNBOOK.md` is the exact
GPU-collection path.

## Three-layer architecture (and the CPU/GPU split that governs the code)

Everything **except GPU collection** runs and is tested on CPU. This is the single
most important constraint in the repo:

> **All torch / transformers / peft imports MUST live inside functions, never at
> module top level.** This keeps `py_compile`, `unittest`, and the `tap/` pipeline
> working on a CPU-only (often ARM64) host with no GPU libraries. `branch.py` and
> the probes follow this; preserve it in any new code.

- **Collection — `math_loop/` (GPU pod).** `tap_controller.py` is the driver: it
  walks `chains × states × candidates`. `branch.py` is the primitive — load a
  byte-identical *before-state* and apply **exactly one** GRPO step via prime-rl,
  writing a raw-artifact tree (`before/`, `state.json`, `cand_<k>/`).
  `tap_probes.py` computes teacher-forced probe NLL + generic-KL drift.
  `features.py` (`convert()`) turns the raw tree into the 4 Parquet tables.
- **Launcher + safety — `run_prime_rl_math_loop.py`, `reap_pods.py`.** Provisions a
  Prime Intellect pod, pins prime-rl, drives collection→featurization on the pod,
  downloads artifacts, schema-validates, runs `tap.run_all`, and guarantees
  teardown. See the safety invariants below — they are load-bearing.
- **Model + eval — `tap/` (CPU).** `dataset.py` loads Parquet → chain splits +
  ablation masks; `model.py` is `SmallTAP` (~110k params, attention over history);
  `baselines.py` has the heuristic selectors + learned baselines; `train.py` fits;
  `eval.py` does within-state ranking metrics; `run_all.py` is the end-to-end entry
  that emits `results.csv` + `report.{md,tex,pdf}`.

**Data flow:** `tap_controller` (raw tree) → `math_loop.features` (4 Parquet) →
`tap.schema --validate` → `tap.run_all` (train + eval + report).

### The "fresh-branch-weights" recipe (don't break the invariants)

Every candidate at a state branches from a **byte-identical** before-state —
`branch.assert_identical_before_state` checks `checkpoint_hash` +
`optimizer_state_hash`. Each branch is one fresh GRPO step (weights-only,
`run_default/` checkpoint layout, **no optimizer resume**). The main chain advances
by a **seeded-random** candidate so the collected history is unbiased. A historical
branch's own utility label is never fed back in as an input feature.

### The frozen schema contract (`tap/schema.py`)

Four Parquet tables (`states`, `trajectories`, `candidates`, `history`) with
**frozen column names, order, dtypes, and vector widths** (e.g. `policy_fingerprint`
32, `candidate_embedding`/`trajectory_embedding` 256, `gradient_sketch` 64).
Field names are transcribed verbatim from the spec — **do not rename, add, drop, or
reorder columns** without updating every producer/consumer. `--validate` enforces
the contract exactly and is the gate the launcher runs before accepting a
collection. `candidates.parquet` is the main training table.

The label (computed in featurization, nats/token):
```
utility_points = 1000 * (0.8*matched_gain + 0.2*global_gain
                         - 0.03*max(incremental_generic_kl, 0))
```

## Commands

### Environment (CPU dev)

The `tap/` pipeline and most tests need `numpy/pandas/pyarrow/scikit-learn` (+
optional CPU torch). `uv` is the package manager; there is **no** committed `.venv`:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv numpy pandas pyarrow scikit-learn
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu
```

Run CPU Python through the venv with `--no-project` (the repo has no
`pyproject.toml`):

```bash
VIRTUAL_ENV=$PWD/.venv uv run --no-project python -m <module>
```

### Tests & static checks

```bash
python -m py_compile run_prime_rl_math_loop.py math_loop/*.py tap/*.py tests/*.py
VIRTUAL_ENV=$PWD/.venv uv run --no-project python -m unittest discover     # full suite
VIRTUAL_ENV=$PWD/.venv uv run --no-project python -m unittest tests.test_tap_engine   # single module
VIRTUAL_ENV=$PWD/.venv uv run --no-project python -m unittest tests.test_tap_engine.FeatureExtractorTests   # single case
```

- `TAP_NO_TORCH=1` forces the sklearn fallback path (gradient sketch zeroed,
  `SmallTAP` replaced); torch-only tests `skipUnless` torch is importable. Run the
  suite both with and without it when touching model/feature code.
- Stdlib-only tests (`test_math_loop`, `test_launcher_teardown`) run under plain
  `python -m unittest` with no venv.

### TAP eval on CPU (synthetic or real)

```bash
# Generate a synthetic dataset (plumbing check only — not a scientific result):
VIRTUAL_ENV=$PWD/.venv uv run --no-project python -m tap.synth --out outputs/tap_synth_192 --labels 192
# End-to-end eval + report:
VIRTUAL_ENV=$PWD/.venv uv run --no-project python -m tap.run_all --parquet-dir outputs/tap_synth_192 --out outputs/tap_report
# Validate a collection against the frozen contract:
VIRTUAL_ENV=$PWD/.venv uv run --no-project python -m tap.schema --validate <parquet_dir>
```

### Collection driver / data prep (no network on `--dry-run`)

```bash
python -m math_loop.tap_controller --dry-run        # plan only, no heavy imports
python -m math_loop.features --raw <raw_root> --out <parquet_dir>
python -m math_loop.data --include-final            # prepare MATH splits (incl. MATH-500 for final eval only)
```

### GPU collection on Prime Intellect (see RUNBOOK.md)

```bash
export PRIME_API_KEY=<key>          # env only — NEVER write it to a tracked file
python run_prime_rl_math_loop.py --dry-run --smoke --prime-rl-commit <sha>   # print plan, no pod
python run_prime_rl_math_loop.py --smoke --provider lambdalabs --gpu-type H100_80GB \
  --prime-rl-commit <sha> --max-cost-usd 40 --ssh-key /workspace/private_key.pem
python reap_pods.py --list          # active tap-v1- pods + count
python reap_pods.py --terminate-prefix tap-v1-smoke-   # reap smoke pods
```

prime-rl itself runs on the pod via `uv run rl` and needs **2 GPUs** (1 trainer + 1
inference). The launcher requires a `--prime-rl-commit` pin.

## Launcher safety invariants (do not weaken)

These are enforced in `run_prime_rl_math_loop.py` / `reap_pods.py` and covered by
`tests/test_launcher_teardown.py`. Treat them as guardrails, not suggestions:

- **The agent loop may only ever launch the cheap `--smoke` run.** Any non-`--smoke`
  run (and `--gpu-count > 1` beyond smoke caps) requires `TAP_ALLOW_FULL_RUN=1`,
  which the loop never sets — only a human launches a full collection.
- **Cost is fail-closed.** A cost monitor (`offer price/h × elapsed`, default cap
  ≤ $80) **and** an independent wall-clock deadline reap the pod and exit non-zero
  on breach *or on the monitor's own failure*. `--keep-pod` is forbidden.
- **Teardown is guaranteed** via atexit + SIGINT/SIGTERM + monitor breach, reaping
  by both the `tap-v1-` name prefix and the id written to `pod_id.txt`.
  `reap_pods.py` refuses any empty or non-`tap-v1-` terminate prefix.
- **Secrets never touch disk.** `PRIME_API_KEY` lives only in the environment; SSH
  uses `/workspace/private_key.pem`. `.gitignore` blocks `*.pem`, `*.key`, `.env`,
  `pod_id.txt`. Keep it that way.

## Docs / reports

`tap/report.py` and the `docs/*/build_*.py` scripts render Markdown + LaTeX, and a
PDF via **reportlab** (the ARM64 build host has no LaTeX toolchain, so a `.tex`
source ships alongside a reportlab-rendered PDF). Match this convention for new
reports rather than assuming `pdflatex` is available.
