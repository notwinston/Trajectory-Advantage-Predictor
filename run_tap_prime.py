#!/usr/bin/env python3
"""Launch TAP v1 fixed-rollout collection/training on a Prime Intellect pod."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import shutil
import subprocess
import sys
import time

from tap_loop.layout import SUBDIRS, default_run_id


DEFAULT_GPU_TYPE = "H100_80GB"
DEFAULT_PROVIDER = "datacrunch"
IMAGE = "ubuntu_22_cuda_12"
REMOTE_REPO = "/workspace/tap_loop_repo"
REMOTE_PRIME_RL = "/workspace/prime-rl"
DEFAULT_DISK_RUN_BASE = Path("/mnt/prime_tap/tap_runs")
DEFAULT_EPHEMERAL_RUN_BASE = Path("/workspace/tap_runs")


def run(command: list[str], *, timeout: int = 300, log: Path | None = None, stream: bool = False) -> str:
    if stream:
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        output: list[str] = []
        log_handle = log.open("a", encoding="utf-8") if log else None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                output.append(line)
                if log_handle:
                    log_handle.write(line)
            returncode = process.wait(timeout=timeout)
        finally:
            if log_handle:
                log_handle.close()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command, "".join(output))
        return "".join(output)

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if log:
        log.write_text(result.stdout, encoding="utf-8")
    elif result.returncode and result.stdout:
        print(result.stdout, file=sys.stderr, end="" if result.stdout.endswith("\n") else "\n")
    result.check_returncode()
    return result.stdout


def prime_json(*args: str) -> dict:
    return json.loads(run(["prime", *args, "--output", "json"]))


def normalize_gpu_type(value: str) -> str:
    return str(value).removesuffix(" (Spot)").replace(" ", "_")


def resource_value(value) -> str:
    match = re.search(r"\d+", str(value))
    if not match:
        raise ValueError(f"could not parse resource value from {value!r}")
    return match.group()


def created_pod_id(output: str) -> str:
    match = re.search(r"Successfully created pod ([\w-]+)", output)
    if not match:
        raise RuntimeError(f"could not parse created pod ID from:\n{output}")
    return match.group(1)


def select_offer(resources: list[dict], args: argparse.Namespace) -> dict:
    matches = []
    for offer in resources:
        if args.cloud_id and offer.get("cloud_id") != args.cloud_id:
            continue
        if str(offer.get("provider", "")).lower() != args.provider.lower():
            continue
        if normalize_gpu_type(offer.get("gpu_type", "")) != args.gpu_type:
            continue
        if int(offer.get("gpu_count", 0)) != args.gpu_count:
            continue
        if bool(offer.get("is_spot")) != args.spot:
            continue
        if str(offer.get("stock_status", "")).lower() != "available":
            continue
        price = offer.get("price_value")
        if args.max_price_per_hour is not None and price is not None and price > args.max_price_per_hour:
            continue
        matches.append(offer)
    if not matches:
        raise ValueError(f"no available {args.gpu_count}x {args.gpu_type} offers for provider={args.provider}")
    return sorted(matches, key=lambda offer: offer.get("price_value") or float("inf"))[0]


def parse_ssh_connection(status: dict) -> tuple[str, str, int]:
    connection = status["ssh"]
    if isinstance(connection, list):
        connection = connection[0]
    tokens = shlex.split(connection)
    user, host = next(token for token in tokens if "@" in token).rsplit("@", 1)
    port = 22
    for index, token in enumerate(tokens):
        if token in {"-p", "--port"}:
            port = int(tokens[index + 1])
            break
    return user, host, port


def ssh_transport(key: Path, port: int) -> list[str]:
    return shlex.split(
        f"ssh -i {shlex.quote(str(key))} -p {port} "
        "-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    )


def wait_for_status(pod_id: str, timeout: int = 1200) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = prime_json("pods", "status", pod_id)
        state = str(status.get("status", "")).upper()
        if state == "ACTIVE" and status.get("ssh"):
            return status
        if status.get("installation_error"):
            raise RuntimeError(status["installation_error"])
        if state in {"ERROR", "FAILED", "TERMINATED", "CANCELLED"}:
            raise RuntimeError(f"pod entered terminal state {state}")
        time.sleep(10)
    raise TimeoutError(f"pod {pod_id} did not become ready within {timeout}s")


def remote(ssh: list[str], destination: str, command: list[str], **kwargs) -> str:
    return run([*ssh, destination, shlex.join(command)], **kwargs)


def terminate_pod(pod_id: str) -> None:
    run(["prime", "pods", "terminate", pod_id, "--yes"])


def preflight_issues(args: argparse.Namespace) -> list[str]:
    issues: list[str] = []
    if shutil.which("prime") is None:
        issues.append(
            "Prime CLI is not installed. Install/auth with: uv tool install prime; prime login; "
            "prime config set-ssh-key-path ~/.ssh/id_rsa"
        )
    if shutil.which("rsync") is None:
        issues.append("rsync is required locally for source upload and artifact mirroring")
    if not args.ephemeral_ok and not args.disk_id:
        issues.append("a persistent --disk-id is required unless --ephemeral-ok is set")
    if args.gpu_count < 3:
        issues.append("--gpu-count must be at least 3 for TAP v1")
    if not args.ssh_key.expanduser().is_file():
        issues.append(f"SSH private key does not exist: {args.ssh_key.expanduser()}")
    if args.backend == "torch" and not (args.hf_token or os.environ.get("HF_TOKEN")):
        issues.append("set --hf-token or HF_TOKEN so Hugging Face auth is propagated to the pod")
    return issues


def remote_run_root(args: argparse.Namespace) -> Path:
    if args.remote_root:
        return args.remote_root
    base = DEFAULT_EPHEMERAL_RUN_BASE if args.ephemeral_ok and not args.disk_id else DEFAULT_DISK_RUN_BASE
    return base / args.run_id


def build_bootstrap_command(args: argparse.Namespace, run_root: Path) -> list[str]:
    subdirs = " ".join(shlex.quote(str(run_root / subdir)) for subdir in SUBDIRS)
    hf_check = 'test -n "${HF_TOKEN:-}"' if args.backend == "torch" else ":"
    script = f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git rsync python3-venv build-essential
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"
{hf_check}
if [ ! -d {REMOTE_PRIME_RL}/.git ]; then
  git clone --depth 1 https://github.com/PrimeIntellect-ai/prime-rl {REMOTE_PRIME_RL}
fi
cd {REMOTE_PRIME_RL}
git submodule update --init deps/verifiers deps/renderers deps/research-environments deps/pydantic-config || true
uv sync --all-extras
uv pip install pyarrow scikit-learn
mkdir -p {shlex.quote(str(run_root))} {subdirs}
test -w {shlex.quote(str(run_root))}
"""
    return ["bash", "-lc", script]


def build_remote_pipeline_command(args: argparse.Namespace, run_root: Path) -> list[str]:
    collector = [
        "uv",
        "run",
        "python",
        "-m",
        "tap_loop.collector",
        "--run-root",
        str(run_root),
        "--chains",
        str(args.chains),
        "--states-per-chain",
        str(args.states_per_chain),
        "--candidates-per-state",
        str(args.candidates_per_state),
        "--batch-prompts",
        str(args.batch_prompts),
        "--group-size",
        str(args.group_size),
        "--max-completion-tokens",
        str(args.max_completion_tokens),
        "--gpu-count",
        str(args.gpu_count),
        "--seed",
        str(args.seed),
        "--backend",
        args.backend,
    ]
    pieces = [
        f"cd {shlex.quote(REMOTE_PRIME_RL)}",
        "export PATH=\"$HOME/.local/bin:/root/.local/bin:$PATH\"",
        f"export PYTHONPATH={shlex.quote(REMOTE_REPO)}:${{PYTHONPATH:-}}",
        f"export HF_HOME={shlex.quote(str(run_root / 'hf_cache'))}",
        shlex.join(collector),
        shlex.join(["uv", "run", "python", "-m", "tap_loop.train_tap", "--run-root", str(run_root)]),
    ]
    return ["bash", "-lc", " && ".join(pieces)]


def upload_repo(ssh: list[str], destination: str, repo_root: Path) -> None:
    run(
        [
            "rsync",
            "-az",
            "--delete",
            "--exclude",
            ".git",
            "--exclude",
            "outputs",
            "--exclude",
            "data/math_loop",
            "-e",
            shlex.join(ssh),
            f"{repo_root}/",
            f"{destination}:{REMOTE_REPO}/",
        ]
    )


def download_outputs(ssh: list[str], destination: str, remote_root: Path, local_root: Path) -> None:
    local_root.mkdir(parents=True, exist_ok=True)
    run(
        [
            "rsync",
            "-az",
            "-e",
            shlex.join(ssh),
            f"{destination}:{remote_root}/",
            f"{local_root}/",
        ]
    )


def remote_pipeline_with_periodic_sync(
    ssh: list[str],
    destination: str,
    command: list[str],
    *,
    timeout: int,
    log: Path,
    remote_root: Path,
    local_root: Path,
    sync_interval_min: int,
) -> None:
    rendered = [*ssh, destination, shlex.join(command)]
    process = subprocess.Popen(rendered, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    sync_interval = max(sync_interval_min, 1) * 60
    next_sync = time.monotonic() + sync_interval
    with log.open("a", encoding="utf-8") as log_handle:
        while True:
            if time.monotonic() > deadline:
                process.kill()
                process.wait()
                raise TimeoutError(f"remote command timed out after {timeout}s")
            for key, _ in selector.select(timeout=5):
                line = key.fileobj.readline()
                if line:
                    sys.stdout.write(line)
                    log_handle.write(line)
            if time.monotonic() >= next_sync:
                try:
                    download_outputs(ssh, destination, remote_root, local_root)
                    log_handle.write(f"[tap-sync] mirrored artifacts at {dt.datetime.now(dt.timezone.utc).isoformat()}\n")
                except Exception as exc:
                    log_handle.write(f"[tap-sync] mirror failed: {exc}\n")
                next_sync = time.monotonic() + sync_interval
            returncode = process.poll()
            if returncode is not None:
                for line in process.stdout:
                    sys.stdout.write(line)
                    log_handle.write(line)
                if returncode:
                    raise subprocess.CalledProcessError(returncode, rendered)
                return


def create_pod(args: argparse.Namespace) -> str:
    if args.offer_id:
        create = ["prime", "pods", "create", "--id", args.offer_id]
    else:
        availability = [
            "availability",
            "list",
            "--gpu-type",
            args.gpu_type,
            "--gpu-count",
            str(args.gpu_count),
            "--provider",
            args.provider,
            "--no-group-similar",
        ]
        if args.disk_id:
            availability.extend(["--disks", args.disk_id])
        resources = prime_json(*availability)["gpu_resources"]
        offer = select_offer(resources, args)
        create = ["prime", "pods", "create", "--id", offer["id"]]
        for option, value in {
            "--disk-size": str(args.disk_size_gb or resource_value(offer["disk_gb"])),
            "--vcpus": resource_value(offer["vcpus"]),
            "--memory": resource_value(offer["memory_gb"]),
        }.items():
            create.extend((option, value))

    now = dt.datetime.now(dt.timezone.utc)
    create.extend(["--name", f"tap-v1-{now:%Y%m%d-%H%M%S}", "--image", IMAGE])
    if args.disk_id:
        create.extend(["--disks", args.disk_id])
    if args.hf_token:
        create.extend(["--env", f"HF_TOKEN={args.hf_token}"])
    create.append("--yes")
    return created_pod_id(run(create))


def write_manifest(local_root: Path, args: argparse.Namespace, pod_id: str | None, run_root: Path) -> None:
    local_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": args.run_id,
        "pod_id": pod_id,
        "remote_root": str(run_root),
        "local_root": str(local_root),
        "gpu_type": args.gpu_type,
        "gpu_count": args.gpu_count,
        "disk_id": args.disk_id,
        "backend": args.backend,
        "states_per_chain": args.states_per_chain,
        "candidates_per_state": args.candidates_per_state,
        "batch_prompts": args.batch_prompts,
        "group_size": args.group_size,
        "max_completion_tokens": args.max_completion_tokens,
    }
    (local_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_on_pod(args: argparse.Namespace, pod_id: str, repo_root: Path, local_root: Path, run_root: Path) -> None:
    print("Waiting for SSH...", flush=True)
    user, host, port = parse_ssh_connection(wait_for_status(pod_id))
    ssh = ssh_transport(args.ssh_key, port)
    destination = f"{user}@{host}"
    log = local_root / "logs" / "pod.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("", encoding="utf-8")

    print("Uploading source...", flush=True)
    remote(ssh, destination, ["mkdir", "-p", REMOTE_REPO, str(run_root)])
    upload_repo(ssh, destination, repo_root)

    print("Bootstrapping remote TAP environment...", flush=True)
    remote(ssh, destination, build_bootstrap_command(args, run_root), timeout=7200, log=log, stream=True)

    print("Running TAP collector...", flush=True)
    try:
        remote_pipeline_with_periodic_sync(
            ssh,
            destination,
            build_remote_pipeline_command(args, run_root),
            timeout=args.remote_timeout,
            log=log,
            remote_root=run_root,
            local_root=local_root,
            sync_interval_min=args.sync_interval_min,
        )
    finally:
        print("Mirroring TAP artifacts...", flush=True)
        download_outputs(ssh, destination, run_root, local_root)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--cloud-id")
    parser.add_argument("--offer-id")
    parser.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE)
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--disk-id")
    parser.add_argument("--remote-root", type=Path)
    parser.add_argument("--ephemeral-ok", action="store_true")
    parser.add_argument("--ssh-key", type=Path, default=Path("~/.ssh/id_rsa"))
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--max-price-per-hour", type=float)
    parser.add_argument("--disk-size-gb", type=int, default=1500)
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--states-per-chain", type=int, default=6)
    parser.add_argument("--candidates-per-state", type=int, default=6)
    parser.add_argument("--batch-prompts", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=192)
    parser.add_argument("--sync-interval-min", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--backend", choices=("torch", "dry-run"), default="torch")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--keep-pod", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--remote-timeout", type=int, default=7 * 24 * 3600)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN", "")
    args.ssh_key = args.ssh_key.expanduser().resolve()
    repo_root = Path(__file__).resolve().parent
    local_root = (args.output_dir or Path("outputs") / "tap_v1" / args.run_id).expanduser().resolve()
    run_root = remote_run_root(args)
    issues = preflight_issues(args)

    if args.dry_run:
        print("Dry run. No pod will be created.")
        if issues:
            print("Preflight issues that would block a real run:")
            for issue in issues:
                print(f"- {issue}")
        print(f"Would upload {repo_root} to {REMOTE_REPO}")
        print(f"Would use remote run root {run_root}")
        print(f"Would mirror artifacts to {local_root}")
        print("Would bootstrap with:")
        print(shlex.join(build_bootstrap_command(args, run_root)))
        print("Would run collector with:")
        print(shlex.join(build_remote_pipeline_command(args, run_root)))
        write_manifest(local_root, args, None, run_root)
        return

    if issues:
        raise SystemExit("Preflight failed:\n" + "\n".join(f"- {issue}" for issue in issues))

    local_root.mkdir(parents=True, exist_ok=True)
    pod_id = create_pod(args)
    write_manifest(local_root, args, pod_id, run_root)
    mirror_ok = False
    try:
        run_on_pod(args, pod_id, repo_root, local_root, run_root)
        mirror_ok = True
    finally:
        if args.keep_pod or not mirror_ok:
            print(f"Keeping pod {pod_id}")
        else:
            print(f"Terminating pod {pod_id}")
            terminate_pod(pod_id)
    print(f"TAP artifacts saved under {local_root}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from None
