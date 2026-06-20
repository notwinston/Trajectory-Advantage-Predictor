#!/usr/bin/env python3
"""Launch the basic prime-rl Qwen3 MATH loop on a PrimeIntellect pod."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


DEFAULT_GPU_TYPE = "H100_80GB"
DEFAULT_PROVIDER = "datacrunch"
IMAGE = "ubuntu_22_cuda_12"
REMOTE_REPO = "/workspace/math_loop_repo"
REMOTE_PRIME_RL = "/workspace/prime-rl"
REMOTE_WORK = "/workspace/math_loop_runs"


def run(command: list[str], *, timeout: int = 300, log: Path | None = None, stream: bool = False) -> str:
    if stream:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
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
        raise ValueError(
            f"no available {args.gpu_count}x {args.gpu_type} offers for provider={args.provider}"
        )
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


def build_bootstrap_command() -> list[str]:
    script = f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git rsync python3-venv build-essential
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"
if [ ! -d {REMOTE_PRIME_RL}/.git ]; then
  git clone --depth 1 https://github.com/PrimeIntellect-ai/prime-rl {REMOTE_PRIME_RL}
fi
cd {REMOTE_PRIME_RL}
uv sync
mkdir -p {REMOTE_WORK}
"""
    return ["bash", "-lc", script]


def build_remote_controller_command(args: argparse.Namespace) -> list[str]:
    probe_device = "cuda:2" if args.gpu_count >= 3 else "cuda"
    controller = [
        "uv",
        "run",
        "python",
        "-m",
        "math_loop.controller",
        "--config",
        REMOTE_REPO + "/configs/prime_rl/qwen3_8b_math_state.toml",
        "--data-dir",
        REMOTE_WORK + "/data",
        "--output-dir",
        REMOTE_WORK + "/outputs",
        "--states",
        str(args.states),
        "--candidates-per-state",
        str(args.candidates_per_state),
        "--batch-prompts",
        str(args.batch_prompts),
        "--group-size",
        str(args.group_size),
        "--gpu-count",
        str(args.gpu_count),
        "--probe-device",
        probe_device,
    ]
    if args.skip_probe_loss:
        controller.append("--skip-probe-loss")
    pieces = [
        f"cd {shlex.quote(REMOTE_PRIME_RL)}",
        "export PATH=\"$HOME/.local/bin:/root/.local/bin:$PATH\"",
        f"export PYTHONPATH={shlex.quote(REMOTE_REPO)}:${{PYTHONPATH:-}}",
        f"export HF_HOME={shlex.quote(REMOTE_WORK + '/hf_cache')}",
        shlex.join(controller),
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


def download_outputs(ssh: list[str], destination: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "rsync",
            "-az",
            "-e",
            shlex.join(ssh),
            f"{destination}:{REMOTE_WORK}/outputs/",
            f"{output_dir}/",
        ]
    )


def run_on_pod(args: argparse.Namespace, pod_id: str, repo_root: Path, output_dir: Path) -> None:
    print("Waiting for SSH...", flush=True)
    user, host, port = parse_ssh_connection(wait_for_status(pod_id))
    ssh = ssh_transport(args.ssh_key, port)
    destination = f"{user}@{host}"
    log = output_dir / "pod.log"
    log.write_text("", encoding="utf-8")

    print("Uploading source...", flush=True)
    remote(ssh, destination, ["mkdir", "-p", REMOTE_REPO, REMOTE_WORK])
    upload_repo(ssh, destination, repo_root)

    print("Installing prime-rl dependencies on the pod...", flush=True)
    remote(ssh, destination, build_bootstrap_command(), timeout=3600, log=log, stream=True)

    print("Running math loop controller...", flush=True)
    remote(ssh, destination, build_remote_controller_command(args), timeout=args.remote_timeout, log=log, stream=True)

    print("Downloading outputs...", flush=True)
    download_outputs(ssh, destination, output_dir)


def create_pod(args: argparse.Namespace) -> str:
    if args.offer_id:
        create = ["prime", "pods", "create", "--id", args.offer_id]
    else:
        resources = prime_json(
            "availability",
            "list",
            "--gpu-type",
            args.gpu_type,
            "--gpu-count",
            str(args.gpu_count),
            "--provider",
            args.provider,
            "--no-group-similar",
        )["gpu_resources"]
        offer = select_offer(resources, args)
        create = ["prime", "pods", "create", "--id", offer["id"]]
        for option, value in {
            "--disk-size": str(args.disk_size_gb or resource_value(offer["disk_gb"])),
            "--vcpus": resource_value(offer["vcpus"]),
            "--memory": resource_value(offer["memory_gb"]),
        }.items():
            create.extend((option, value))

    now = dt.datetime.now(dt.timezone.utc)
    create.extend(
        [
            "--name",
            f"qwen3-math-loop-{now:%Y%m%d-%H%M%S}",
            "--image",
            IMAGE,
            "--yes",
        ]
    )
    return created_pod_id(run(create))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--cloud-id")
    parser.add_argument("--offer-id", help="skip offer selection and create this exact Prime offer id")
    parser.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE)
    parser.add_argument("--gpu-count", type=int, default=2)
    parser.add_argument("--ssh-key", type=Path, default=Path("~/.ssh/id_rsa"))
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--max-price-per-hour", type=float)
    parser.add_argument("--disk-size-gb", type=int, default=1500)
    parser.add_argument("--states", type=int, default=48)
    parser.add_argument("--candidates-per-state", type=int, default=16)
    parser.add_argument("--batch-prompts", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--skip-probe-loss", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen3_math_loop"))
    parser.add_argument("--keep-pod", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--remote-timeout", type=int, default=7 * 24 * 3600)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    args.ssh_key = args.ssh_key.expanduser().resolve()
    if args.gpu_count < 2:
        raise SystemExit("--gpu-count must be at least 2")
    if not args.dry_run and not args.ssh_key.is_file():
        raise SystemExit(f"SSH private key does not exist: {args.ssh_key}")
    if args.max_price_per_hour is not None and args.max_price_per_hour <= 0:
        raise SystemExit("--max-price-per-hour must be positive")

    if args.dry_run:
        print("Dry run. No pod will be created.")
        print(f"Would upload {repo_root} to {REMOTE_REPO}")
        print("Would bootstrap prime-rl with:")
        print(shlex.join(build_bootstrap_command()))
        print("Would run controller with:")
        print(shlex.join(build_remote_controller_command(args)))
        return

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pod_id = create_pod(args)
    try:
        run_on_pod(args, pod_id, repo_root, output_dir)
    finally:
        if args.keep_pod:
            print(f"Keeping pod {pod_id}")
        else:
            print(f"Terminating pod {pod_id}")
            terminate_pod(pod_id)
    print(f"Outputs saved under {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from None
