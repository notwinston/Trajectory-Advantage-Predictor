#!/usr/bin/env python3
"""Launch the TAP GRPO battery on a single-GPU Prime Intellect pod.

Lightweight (no prime-rl/vLLM/flash-attn): just torch/transformers/peft/datasets +
lightgbm/shap, then ``python -m tap.battery``. Reuses pod/SSH plumbing from
``run_prime_rl_math_loop`` (offer selection, upload, download, terminate).

Examples:
  # noise gate (cheap): few cohorts x seeds, small probe
  python run_tap_pod.py --gpu-count 1 --max-cohorts 4 --seeds 4 \
      --probe-size 24 --grpo-steps 4 --output-dir outputs/tap_noise
  # full battery with chains (for leave-one-chain-out)
  python run_tap_pod.py --gpu-count 1 --n-chains 3 --anchors-per-chain 4 \
      --output-dir outputs/tap_full
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import time

import run_prime_rl_math_loop as base


def _wait_for_ssh(ssh, dest, *, attempts: int = 15, delay: int = 15) -> None:
    """Poll SSH until cloud-init has injected the key (avoids a provisioning race)."""

    for i in range(attempts):
        try:
            base.remote(ssh, dest, ["true"], timeout=30)
            return
        except subprocess.CalledProcessError:
            print(f"ssh not ready ({i + 1}/{attempts}); waiting {delay}s...", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"ssh to {dest} never became ready")

REMOTE_REPO = base.REMOTE_REPO
REMOTE_WORK = base.REMOTE_WORK


def build_bootstrap_command() -> list[str]:
    script = f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt_retry() {{ for i in $(seq 1 60); do if apt-get -o DPkg::Lock::Timeout=600 "$@"; then return 0; fi; echo "apt retry $i"; sleep 10; done; return 1; }}
apt_retry update
apt_retry install -y --no-install-recommends ca-certificates curl git rsync python3-venv
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"
mkdir -p {REMOTE_WORK} && cd {REMOTE_WORK}
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install torch transformers peft datasets accelerate lightgbm shap
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"""
    return ["bash", "-lc", script]


def build_battery_command(args: argparse.Namespace) -> list[str]:
    common = ["python", "-m", "tap.battery",
              "--data-dir", REMOTE_WORK + "/data",
              "--model-name", args.model_name,
              "--grpo-steps", str(args.grpo_steps),
              "--group-size", str(args.group_size),
              "--temperature", str(args.temperature),
              "--max-new-tokens", str(args.max_new_tokens),
              "--probe-size", str(args.probe_size),
              "--probe-k", str(args.probe_k),
              "--cohort-size", str(args.cohort_size),
              "--n-random", str(args.n_random),
              "--n-chains", str(args.n_chains),
              "--anchors-per-chain", str(args.anchors_per_chain),
              "--seeds", str(args.seeds),
              "--seed", str(args.seed),
              "--lr", str(args.lr)]
    if args.max_cohorts:
        common += ["--max-cohorts", str(args.max_cohorts)]
    if not args.acc_eval:
        common += ["--no-acc-eval"]
    out = REMOTE_WORK + "/outputs"
    env = [f"cd {shlex.quote(REMOTE_WORK)}", ". .venv/bin/activate",
           'export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"',
           f"export PYTHONPATH={shlex.quote(REMOTE_REPO)}:${{PYTHONPATH:-}}",
           f"export HF_HOME={shlex.quote(REMOTE_WORK + '/hf_cache')}",
           f"mkdir -p {shlex.quote(out)}"]  # parallel shards redirect into outputs/ before python runs

    G = max(1, args.gpu_count)
    total = args.shard_total or G          # global #shards across all pods
    base = args.shard_base                 # this pod's first global shard index
    if total > 1:
        # warm the data cache ONCE so the G parallel shard processes don't race on split creation
        pycode = ('from pathlib import Path; from tap.data_probes import prepare_probes; '
                  f'prepare_probes(Path("{REMOTE_WORK}/data"), probe_size={args.probe_size})')
        prewarm = "python -c " + shlex.quote(pycode)
        cstr = shlex.join(common)
        loop = (f'pids=""; for i in $(seq 0 {G - 1}); do s=$(({base}+i)); '
                f'CUDA_VISIBLE_DEVICES=$i {cstr} --shard $s/{total} '
                f'--output {out}/labels_shard_$s.jsonl > {out}/shard_$s.log 2>&1 & pids="$pids $!"; done; '
                f'for p in $pids; do wait $p; done; '
                f'cat {out}/labels_shard_*.jsonl > {out}/labels.jsonl; wc -l {out}/labels.jsonl')
        pieces = env + [prewarm, loop]
    else:
        pieces = env + [shlex.join(common + ["--output", out + "/labels.jsonl"])]
    return ["bash", "-lc", " && ".join(pieces)]


def run_on_pod(args, pod_id, repo_root, output_dir, *, reuse=None) -> None:
    if reuse is not None:  # reuse an existing pod: skip create + bootstrap, just refresh code + run
        dest, port = reuse
        ssh = base.ssh_transport(args.ssh_key, port)
        print(f"Reusing pod {dest} (skip create/deps)...", flush=True)
    else:
        print("Waiting for SSH...", flush=True)
        user, host, port = base.parse_ssh_connection(base.wait_for_status(pod_id))
        ssh = base.ssh_transport(args.ssh_key, port)
        dest = f"{user}@{host}"
    log = output_dir / "pod.log"
    log.write_text("", encoding="utf-8")
    _wait_for_ssh(ssh, dest)
    base.remote(ssh, dest, ["mkdir", "-p", REMOTE_REPO, REMOTE_WORK])
    base.upload_repo(ssh, dest, repo_root)
    if reuse is None:
        print("Installing deps...", flush=True)
        base.remote(ssh, dest, build_bootstrap_command(), timeout=3600, log=log, stream=True)
    else:  # free the GPU from any prior battery before relaunching
        # [t]ap.battery: regex matches the real process but NOT this pkill's own
        # command line, which would otherwise SIGKILL the ssh shell (rc 255).
        base.remote(ssh, dest, ["bash", "-lc", "pkill -9 -f '[t]ap.battery' || true; sleep 3"])
    print("Running TAP battery...", flush=True)
    base.remote(ssh, dest, build_battery_command(args), timeout=args.remote_timeout, log=log, stream=True)
    print("Downloading outputs...", flush=True)
    base.download_outputs(ssh, dest, output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", default=base.DEFAULT_PROVIDER)
    p.add_argument("--cloud-id")
    p.add_argument("--offer-id")
    p.add_argument("--gpu-type", default=base.DEFAULT_GPU_TYPE)
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--shard-base", type=int, default=0, help="this pod's first global shard index")
    p.add_argument("--shard-total", type=int, default=0, help="total shards across ALL pods (0 => gpu-count)")
    p.add_argument("--ssh-key", type=Path, default=Path("~/.ssh/id_ed25519"))
    p.add_argument("--spot", action="store_true")
    p.add_argument("--max-price-per-hour", type=float)
    p.add_argument("--disk-size-gb", type=int, default=200)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--grpo-steps", type=int, default=4)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--probe-size", type=int, default=64)
    p.add_argument("--probe-k", type=int, default=4)
    p.add_argument("--cohort-size", type=int, default=8)
    p.add_argument("--n-random", type=int, default=8)
    p.add_argument("--n-chains", type=int, default=1)
    p.add_argument("--anchors-per-chain", type=int, default=1)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed", type=int, default=0, help="cohort-generation RNG seed (explicit, avoids --seeds prefix clash)")
    p.add_argument("--max-cohorts", type=int, default=0)
    p.add_argument("--no-acc-eval", dest="acc_eval", action="store_false",
                   help="NLL-only gate (skip slow greedy accuracy eval)")
    p.set_defaults(acc_eval=True)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/tap"))
    p.add_argument("--reuse-host", default=None,
                   help="user@host (or host) of an existing pod to reuse: skip create+bootstrap, keep pod alive")
    p.add_argument("--reuse-port", type=int, default=22)
    p.add_argument("--keep-pod", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--remote-timeout", type=int, default=24 * 3600)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    args.ssh_key = args.ssh_key.expanduser().resolve()
    if args.dry_run:
        print("Dry run. No pod created.")
        print("bootstrap:\n" + shlex.join(build_bootstrap_command()))
        print("battery:\n" + shlex.join(build_battery_command(args)))
        return
    if not args.ssh_key.is_file():
        raise SystemExit(f"SSH key not found: {args.ssh_key}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_host:  # reuse path: never auto-terminate (iterate cheaply on one pod)
        dest = args.reuse_host if "@" in args.reuse_host else f"root@{args.reuse_host}"
        run_on_pod(args, None, repo_root, output_dir, reuse=(dest, args.reuse_port))
        print(f"Outputs under {output_dir} (pod {dest} kept alive — terminate manually when done)")
        return
    pod_id = base.create_pod(args)
    try:
        run_on_pod(args, pod_id, repo_root, output_dir)
    finally:
        if not args.keep_pod:
            print(f"Terminating pod {pod_id}")
            base.terminate_pod(pod_id)
    print(f"Outputs under {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from None
