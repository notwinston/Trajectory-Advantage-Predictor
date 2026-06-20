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

The pod runner reads `~/.prime/config.json` for `ssh_key_path` when `--ssh-key`
is omitted, then falls back to `~/.ssh/quack_prime`, `~/.ssh/id_ed25519`, and
`~/.ssh/id_rsa`.
