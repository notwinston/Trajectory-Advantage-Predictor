#!/usr/bin/env python3
"""Profile one QuACK TopK launch on an ephemeral Prime Intellect B200 pod."""

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


GPU_TYPE = "B200_180GB"
DEFAULT_PROVIDER = "datacrunch"
IMAGE = "ubuntu_22_cuda_12"
REMOTE_REPO = "/workspace/quack"
REMOTE_PROFILE = "/workspace/profile"
NCU_SECTIONS = (
    "LaunchStats",
    "Occupancy",
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "SchedulerStats",
    "WarpStateStats",
)


def run(command, *, timeout=300, log=None, stream=False):
    if stream:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        tee = subprocess.Popen(["tee", "-a", str(log)], text=True, stdin=process.stdout)
        process.stdout.close()
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        finally:
            tee.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
        return ""
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if log:
        log.write_text(result.stdout)
    elif result.returncode and result.stdout:
        print(result.stdout, file=sys.stderr, end="" if result.stdout.endswith("\n") else "\n")
    result.check_returncode()
    return result.stdout


def prime_json(*args):
    return json.loads(run(["prime", *args, "--output", "json"]))


def select_offer(resources, provider, cloud_id, price_cap, spot=False):
    matches = [
        offer
        for offer in resources
        if offer.get("cloud_id") == cloud_id
        and str(offer.get("provider", "")).lower() == provider.lower()
        and str(offer.get("gpu_type", "")).removesuffix(" (Spot)").replace(" ", "_") == GPU_TYPE
        and offer.get("gpu_count") == 1
        and bool(offer.get("is_spot")) == spot
        and str(offer.get("stock_status", "")).lower() == "available"
    ]
    if len(matches) != 1:
        market = "spot" if spot else "on-demand"
        raise ValueError(
            f"expected one available {market} {GPU_TYPE} offer for "
            f"{provider}/{cloud_id}, found {len(matches)}"
        )
    offer = matches[0]
    price = offer.get("price_value")
    if price_cap is not None and price is not None and price > price_cap:
        raise ValueError(f"offer costs ${price:.4f}/hr, above the price limit")
    return offer


def resource_value(value):
    return re.search(r"\d+", str(value)).group()


def created_pod_id(output):
    match = re.search(r"Successfully created pod ([\w-]+)", output)
    if not match:
        raise RuntimeError(f"could not parse created pod ID from:\n{output}")
    return match.group(1)


def parse_ssh_connection(status):
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


def ssh_transport(key, port):
    return shlex.split(
        f"ssh -i {shlex.quote(str(key))} -p {port} "
        "-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    )


def build_bootstrap_command():
    script = f"""
set -e
export DEBIAN_FRONTEND=noninteractive
export PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/bin:/bin
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl python3-venv
if ! command -v ncu >/dev/null; then
  repo=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64
  curl -fsSLo /tmp/cuda-keyring.deb "$repo/cuda-keyring_1.1-1_all.deb"
  dpkg -i /tmp/cuda-keyring.deb
  apt-get update
  apt-get install -y --no-install-recommends cuda-nsight-compute-12-9
fi
python3 -m venv /opt/venv
/opt/venv/bin/pip install -U pip
/opt/venv/bin/pip install -e {REMOTE_REPO}
"""
    return ["bash", "-lc", script]


def build_profile_command(args):
    command = shlex.split(
        f"env PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/bin:/bin "
        f"PYTHONPATH={REMOTE_REPO} ncu --profile-from-start off "
        f"--target-processes all --force-overwrite -o {REMOTE_PROFILE}/topk"
    )
    if args.full:
        command.extend(("--set", "full"))
    else:
        for section in NCU_SECTIONS:
            command.extend(("--section", section))
    command.extend(
        shlex.split(
            f"/opt/venv/bin/python {REMOTE_REPO}/benchmarks/benchmark_topk.py "
            f"--profile --M {args.M} --N {args.N} --k {args.k} --dtype {args.dtype}"
        )
    )
    if args.softmax:
        command.append("--softmax")
    return command


def build_ncu_check_command():
    code = "import torch; x=torch.ones(1,device='cuda'); x.add_(1); torch.cuda.synchronize()"
    return shlex.split(
        "env PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/bin:/bin "
        "ncu --section SpeedOfLight --launch-count 1 --target-processes all "
        "--force-overwrite -o /tmp/quack-ncu-check /opt/venv/bin/python -c "
        + shlex.quote(code)
    )


def terminate_pod(pod_id, command=run, sleep=time.sleep):
    for attempt in range(3):
        try:
            command(["prime", "pods", "terminate", pod_id, "--yes"])
            return
        except Exception:
            if attempt == 2:
                raise
            sleep(5)


def run_managed(pod_id, action, terminate=terminate_pod):
    try:
        return action()
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            terminate(pod_id)
        except Exception as exc:
            if not active_error:
                raise
            print(f"WARNING: failed to terminate pod {pod_id}: {exc}", file=sys.stderr)


def wait_for_status(pod_id, timeout=900):
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


def remote(ssh, destination, command, **kwargs):
    return run([*ssh, destination, shlex.join(command)], **kwargs)


def profile_pod(args, pod_id, repo_root, output_dir):
    print("Waiting for SSH...", flush=True)
    user, host, port = parse_ssh_connection(wait_for_status(pod_id))
    ssh = ssh_transport(args.ssh_key, port)
    destination = f"{user}@{host}"
    remote(ssh, destination, ["mkdir", "-p", REMOTE_REPO, REMOTE_PROFILE])
    print("Uploading source...", flush=True)
    run(
        [
            "rsync",
            "-az",
            "--delete",
            f"--exclude-from={repo_root / '.dockerignore'}",
            "-e",
            shlex.join(ssh),
            f"{repo_root}/",
            f"{destination}:{REMOTE_REPO}/",
        ]
    )
    log = output_dir / "profile.log"
    log.write_text("")
    print("Installing dependencies (this may take several minutes)...", flush=True)
    remote(ssh, destination, build_bootstrap_command(), timeout=1800, log=log, stream=True)
    print("Testing NCU counters...", flush=True)
    try:
        remote(ssh, destination, build_ncu_check_command(), timeout=120, log=log, stream=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"NCU counters are unavailable; see {log}") from exc
    print("Profiling one TopK launch...", flush=True)
    remote(ssh, destination, build_profile_command(args), timeout=1800, log=log, stream=True)
    print("Downloading report...", flush=True)
    run(
        [
            "rsync",
            "-az",
            "-e",
            shlex.join(ssh),
            f"{destination}:{REMOTE_PROFILE}/topk.ncu-rep",
            str(output_dir / "topk.ncu-rep"),
        ]
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--cloud-id", required=True)
    parser.add_argument("--ssh-key", required=True, type=Path)
    parser.add_argument("--M", "--m", dest="M", type=int, default=65536)
    parser.add_argument("--N", "--n", dest="N", type=int, default=1024)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--softmax", action="store_true")
    parser.add_argument("--full", action="store_true", help="collect NCU's full metric set")
    parser.add_argument("--spot", action="store_true", help="require a spot offer")
    parser.add_argument("--max-price-per-hour", type=float)
    parser.add_argument("--output-dir", type=Path, default=Path("profiles"))
    return parser.parse_args(argv)


def main():
    args = parse_args()
    args.ssh_key = args.ssh_key.expanduser().resolve()
    if not args.ssh_key.is_file():
        raise SystemExit(f"SSH private key does not exist: {args.ssh_key}")
    if args.max_price_per_hour is not None and args.max_price_per_hour <= 0:
        raise SystemExit("--max-price-per-hour must be positive")
    if args.M <= 0:
        raise SystemExit("M must be positive")
    if args.N <= 0 or args.N > 4096 or args.N & (args.N - 1):
        raise SystemExit("N must be a power of two in [1, 4096]")
    if args.k <= 0 or args.k > min(args.N, 128) or args.k & (args.k - 1):
        raise SystemExit("k must be a power of two no larger than min(N, 128)")

    resources = prime_json(
        "availability",
        "list",
        "--gpu-type",
        GPU_TYPE,
        "--gpu-count",
        "1",
        "--provider",
        args.provider,
        "--no-group-similar",
    )["gpu_resources"]
    offer = select_offer(
        resources, args.provider, args.cloud_id, args.max_price_per_hour, args.spot
    )
    now = dt.datetime.now(dt.timezone.utc)
    name = f"quack-topk-{now:%Y%m%d-%H%M%S}"
    output_dir = args.output_dir.expanduser().resolve() / f"{now:%Y%m%dT%H%M%SZ}"
    output_dir.mkdir(parents=True, exist_ok=True)
    create = ["prime", "pods", "create", "--id", offer["id"]]
    for option, value in {
        "--name": name,
        "--disk-size": resource_value(offer["disk_gb"]),
        "--vcpus": resource_value(offer["vcpus"]),
        "--memory": resource_value(offer["memory_gb"]),
        "--image": IMAGE,
    }.items():
        create.extend((option, value))
    print(f"Creating {args.provider}/{args.cloud_id} pod...", flush=True)
    output = run([*create, "--yes"])
    pod_id = created_pod_id(output)
    repo_root = Path(__file__).resolve().parents[1]

    def profile():
        pods = prime_json("pods", "list")["pods"]
        if not any(pod.get("id") == pod_id for pod in pods):
            raise RuntimeError(f"created pod {pod_id} was not returned by `prime pods list`")
        profile_pod(args, pod_id, repo_root, output_dir)

    run_managed(pod_id, profile)
    print(f"Profile saved to {output_dir / 'topk.ncu-rep'}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from None
