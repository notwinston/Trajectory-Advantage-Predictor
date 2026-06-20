#!/usr/bin/env python3
"""Launch the prime-rl Qwen3 MATH / TAP data-collection loop on a Prime Intellect pod.

Wave 2 hardening goals (TAP v1):
  * provider defaults to lambdalabs and the pod runs the prime-rl image,
  * --prime-rl-commit is REQUIRED for any real launch (the smoke pins the same
    commit it validated; the full run reuses that pin),
  * a fail-closed cost monitor (offer price x elapsed) AND an independent
    absolute wall-clock deadline tear the pod down and exit non-zero on breach
    OR on the monitor's own failure -- it never merely warns, never fails open,
  * pods are named with a `tap-v1-smoke-` prefix; pod_id.txt is written on
    create; both the atexit handler and the monitor's teardown reap by prefix
    (reaping in-flight pods provisioned before pod_id.txt exists) AND by id,
  * SIGINT/SIGTERM are trapped and trigger the same teardown,
  * --keep-pod is forbidden and --gpu-count > 1 is forbidden unless the
    TAP_ALLOW_FULL_RUN=1 env gate is set (the loop never sets it), so this
    script can only ever launch the cheap 1xH100 --smoke run on its own,
  * the SSH key defaults to the user-provided /workspace/private_key.pem
    (chmod 600 + verified against /workspace/public_key.pem), never a fresh key.

Only torch/transformers/peft-free standard library is imported at module load.
"""

from __future__ import annotations

import argparse
import atexit
import datetime as dt
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Wave-2 constants / safety defaults
# ---------------------------------------------------------------------------
DEFAULT_GPU_TYPE = "H100_80GB"
DEFAULT_PROVIDER = "lambdalabs"
# prime-rl runs on a CUDA 12 base image; prime-rl itself is cloned, pinned to
# --prime-rl-commit and `uv sync`-ed on the pod (see build_bootstrap_command).
PRIME_RL_IMAGE = "ubuntu_22_cuda_12"

# Pods this wave provisions are always smoke pods.
SMOKE_POD_PREFIX = "tap-v1-smoke-"
# reap_pods.py refuses any prefix that does not start with this wave guard.
WAVE_POD_PREFIX = "tap-v1-"

DEFAULT_SSH_KEY = Path("/workspace/private_key.pem")
DEFAULT_PUBLIC_KEY = Path("/workspace/public_key.pem")
DEFAULT_MAX_COST_USD = 80.0
# Independent wall-clock ceilings (seconds). The smoke is tiny; a real (gated)
# run gets a wider, but still bounded, window.
SMOKE_WALLCLOCK_SECONDS = 60 * 60
FULL_WALLCLOCK_SECONDS = 6 * 60 * 60

PRIME_RL_REPO_URL = "https://github.com/PrimeIntellect-ai/prime-rl"

REMOTE_REPO = "/workspace/math_loop_repo"
REMOTE_PRIME_RL = "/workspace/prime-rl"
REMOTE_WORK = "/workspace/math_loop_runs"

REAP_SCRIPT = Path(__file__).resolve().parent / "reap_pods.py"
POD_ID_FILE = Path(__file__).resolve().parent / "pod_id.txt"
DEFAULT_RUNBOOK = Path(__file__).resolve().parent / "RUNBOOK.md"

EXIT_COST_BREACH = 17
EXIT_WALLCLOCK_BREACH = 18
EXIT_MONITOR_FAILURE = 19
EXIT_SIGNAL = 20


# ---------------------------------------------------------------------------
# Low-level subprocess helpers (preserved from the original launcher)
# ---------------------------------------------------------------------------
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
    """Parse `prime ... --output json`, tolerating non-JSON warning prefixes the
    CLI sometimes prints before the payload (e.g. availability auth notices)."""
    output = run(["prime", *args, "--output", "json"])
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        decoder = json.JSONDecoder()
        for index, char in enumerate(output):
            if char not in "{[":
                continue
            try:
                parsed, end = decoder.raw_decode(output[index:])
            except json.JSONDecodeError:
                continue
            if output[index + end:].strip():
                continue
            prefix = output[:index].strip()
            if prefix:
                print(prefix, file=sys.stderr)
            return parsed
        tail = output[-2000:].strip()
        raise RuntimeError(f"Prime CLI did not return parseable JSON. Output tail:\n{tail}") from exc


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


# ---------------------------------------------------------------------------
# prime-rl commit pin
# ---------------------------------------------------------------------------
def resolve_prime_rl_head(timeout: int = 20) -> str | None:
    """Best-effort resolve of prime-rl main HEAD (used only to enrich the
    dry-run plan; a real launch must pass --prime-rl-commit explicitly)."""
    try:
        out = run(["git", "ls-remote", PRIME_RL_REPO_URL, "HEAD"], timeout=timeout)
    except Exception:
        return None
    match = re.search(r"^([0-9a-f]{40})\s", out)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Teardown / reaping (fail-closed)
# ---------------------------------------------------------------------------
class _TeardownState:
    """Process-global teardown coordinates, mutated as the launch progresses."""

    pod_id: str | None = None
    prefix: str = SMOKE_POD_PREFIX
    runbook: Path = DEFAULT_RUNBOOK
    done: bool = False
    lock = threading.Lock()


def _write_banner(runbook: Path, kind: str, detail: str) -> None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    banner = (
        f"\n\n<!-- {kind} -->\n"
        f"## {kind} ({stamp})\n\n"
        f"{detail}\n\n"
        f"The cost/wall-clock monitor reaped the pod and exited non-zero.\n"
    )
    try:
        with runbook.open("a", encoding="utf-8") as handle:
            handle.write(banner)
    except Exception as exc:  # never let logging block teardown
        print(f"[teardown] failed to write banner to {runbook}: {exc!r}", file=sys.stderr)


# Indirection so the SIGTERM teardown unit test can substitute a recorder
# without invoking the real prime CLI.
def _reap_prefix(prefix: str) -> None:
    if not prefix.startswith(WAVE_POD_PREFIX):
        print(f"[teardown] refusing unsafe prefix {prefix!r}", file=sys.stderr)
        return
    try:
        run([sys.executable, str(REAP_SCRIPT), "--terminate-prefix", prefix], timeout=300)
    except Exception as exc:
        print(f"[teardown] reap-by-prefix {prefix!r} failed: {exc!r}", file=sys.stderr)


def _terminate_id(pod_id: str) -> None:
    try:
        run([sys.executable, str(REAP_SCRIPT), "--terminate-id", pod_id], timeout=300)
    except Exception as exc:
        print(f"[teardown] terminate-id {pod_id!r} failed: {exc!r}", file=sys.stderr)


def teardown(reason: str | None = None) -> None:
    """Idempotent teardown: reap by prefix (covers in-flight pods provisioned
    before pod_id.txt is written) AND terminate the known pod id."""
    with _TeardownState.lock:
        if _TeardownState.done:
            return
        _TeardownState.done = True
    if reason:
        print(f"[teardown] {reason}", flush=True)
    # Prefix reap first: catches a pod created during provisioning even if its
    # id never reached pod_id.txt.
    _reap_prefix(_TeardownState.prefix)
    pod_id = _TeardownState.pod_id
    if pod_id is None and POD_ID_FILE.exists():
        try:
            pod_id = POD_ID_FILE.read_text(encoding="utf-8").strip() or None
        except Exception:
            pod_id = None
    if pod_id:
        _terminate_id(pod_id)
    try:
        if POD_ID_FILE.exists():
            POD_ID_FILE.unlink()
    except Exception:
        pass


def _signal_handler(signum, _frame):
    teardown(reason=f"signal {signal.Signals(signum).name} received")
    os._exit(EXIT_SIGNAL)


def install_safety_handlers() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(teardown)


# ---------------------------------------------------------------------------
# Fail-closed cost + wall-clock monitor
# ---------------------------------------------------------------------------
class CostWallclockMonitor(threading.Thread):
    """Independent thread that tears the pod down and hard-exits non-zero on:
      * estimated cost (price/hour x elapsed) >= max_cost_usd,
      * elapsed wall-clock >= max_wallclock_seconds,
      * any internal failure (missing price, exception, etc.) -- fail closed.
    """

    def __init__(self, price_per_hour, max_cost_usd: float, max_wallclock_seconds: float,
                 runbook: Path, poll_seconds: float = 15.0):
        super().__init__(name="tap-cost-monitor", daemon=True)
        self.price_per_hour = price_per_hour
        self.max_cost_usd = max_cost_usd
        self.max_wallclock_seconds = max_wallclock_seconds
        self.runbook = runbook
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._start_monotonic = time.monotonic()

    def stop(self) -> None:
        self._stop.set()

    def _breach(self, kind: str, exit_code: int, detail: str) -> None:
        _write_banner(self.runbook, kind, detail)
        try:
            teardown(reason=f"{kind}: {detail}")
        finally:
            os._exit(exit_code)

    def run(self) -> None:
        try:
            # Fail closed if we cannot price the pod at all.
            if self.price_per_hour is None or float(self.price_per_hour) <= 0:
                self._breach(
                    "COST-CAP-HIT", EXIT_MONITOR_FAILURE,
                    f"offer price unavailable ({self.price_per_hour!r}); cannot enforce "
                    f"${self.max_cost_usd:.0f} cap, failing closed.",
                )
            price = float(self.price_per_hour)
            while not self._stop.is_set():
                elapsed = time.monotonic() - self._start_monotonic
                est_cost = price * (elapsed / 3600.0)
                if est_cost >= self.max_cost_usd:
                    self._breach(
                        "COST-CAP-HIT", EXIT_COST_BREACH,
                        f"estimated spend ${est_cost:.2f} reached cap ${self.max_cost_usd:.2f} "
                        f"(price ${price:.2f}/h x {elapsed/3600:.2f}h).",
                    )
                if elapsed >= self.max_wallclock_seconds:
                    self._breach(
                        "WALLCLOCK-HIT", EXIT_WALLCLOCK_BREACH,
                        f"elapsed {elapsed/3600:.2f}h reached wall-clock deadline "
                        f"{self.max_wallclock_seconds/3600:.2f}h.",
                    )
                self._stop.wait(self.poll_seconds)
        except BaseException as exc:  # noqa: BLE001 - fail closed on ANY error
            self._breach(
                "MONITOR-FAILURE", EXIT_MONITOR_FAILURE,
                f"cost/wall-clock monitor raised {exc!r}; failing closed.",
            )


# ---------------------------------------------------------------------------
# Pod bootstrap / controller commands
# ---------------------------------------------------------------------------
def build_bootstrap_command(prime_rl_commit: str) -> list[str]:
    # Pod-validated bootstrap (apt-lock retry, prime-rl submodules incl. verifiers,
    # uv sync --all-extras, https rewrite for submodule clones) + the wave's
    # explicit --prime-rl-commit pin.
    script = f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
# Pods differ: datacrunch boots as root, lambdalabs as a sudo-capable user with
# no pre-existing /workspace. Handle both.
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; else SUDO=""; fi
$SUDO mkdir -p /workspace
$SUDO chown "$(id -un):$(id -gn)" /workspace 2>/dev/null || true
apt_retry() {{
  for attempt in $(seq 1 60); do
    $SUDO apt-get -o DPkg::Lock::Timeout=30 "$@" && return 0
    status=$?
    echo "apt-get $* failed with status $status; waiting for package lock ($attempt/60)"
    sleep 10
  done
  return "$status"
}}
apt_retry update
apt_retry install -y --no-install-recommends ca-certificates curl git openssh-client rsync python3-venv build-essential
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0="url.https://github.com/.insteadOf"
export GIT_CONFIG_VALUE_0="git@github.com:"
if [ ! -d {REMOTE_PRIME_RL}/.git ]; then
  git clone --recurse-submodules {PRIME_RL_REPO_URL} {REMOTE_PRIME_RL}
fi
cd {REMOTE_PRIME_RL}
git fetch --all --tags
git checkout {shlex.quote(prime_rl_commit)}
git submodule update --init --recursive
git rev-parse HEAD
uv sync --all-extras
# peft is required by the TAP probes (load prime-rl's separately-saved LoRA
# adapter via PeftModel) but is not a prime-rl dependency, so add it explicitly.
uv pip install peft
mkdir -p {REMOTE_WORK}
"""
    return ["bash", "-lc", script]


def _remote_env_prefix() -> list[str]:
    return [
        f"cd {shlex.quote(REMOTE_PRIME_RL)}",
        'export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"',
        f"export PYTHONPATH={shlex.quote(REMOTE_REPO)}:${{PYTHONPATH:-}}",
        f"export HF_HOME={shlex.quote(REMOTE_WORK + '/hf_cache')}",
        "export WANDB_MODE=disabled",
        "export WANDB_DISABLED=true",
    ]


# Unknown (d) from the plan -- NOT feature-degradable. We import the two
# prime-rl symbols the engine relies on; if the pin's API does not match, this
# fails non-zero BEFORE any rollout spend so the smoke can be aborted + the pod
# reaped. The other 3 unknowns carry W1a in-engine fallbacks.
_PREFLIGHT_PY = (
    "import verifiers\n"
    "from prime_rl.orchestrator.advantage import AdvantageOutputs\n"
    "print('preflight OK: verifiers=%s AdvantageOutputs=%s'"
    " % (getattr(verifiers,'__version__','?'), AdvantageOutputs.__name__))\n"
)


def build_preflight_command() -> list[str]:
    pieces = _remote_env_prefix() + ["uv run python -c " + shlex.quote(_PREFLIGHT_PY)]
    return ["bash", "-lc", " && ".join(pieces)]


def remote_run_dir(args: argparse.Namespace) -> str:
    return f"{REMOTE_WORK}/outputs/tap/{args.run_id}"


def build_collection_command(args: argparse.Namespace) -> list[str]:
    """Drive the TAP v1 collection orchestrator then convert raw artifacts ->
    the 4 Parquet. tap_controller owns chains/states/candidates + the raw
    before/ cand_<k>/ state.json tree that math_loop.features consumes."""
    run_dir = remote_run_dir(args)
    collect = [
        "uv", "run", "python", "-m", "math_loop.tap_controller",
        "--run-id", args.run_id,
        "--output-dir", f"{REMOTE_WORK}/outputs/tap",
        "--data-dir", f"{REMOTE_WORK}/data",
        "--chains", str(args.chains),
        "--states-per-chain", str(args.states),
        "--candidates-per-state", str(args.candidates_per_state),
        "--prompts-per-candidate", str(args.prompts_per_candidate),
        "--completions-per-prompt", str(args.completions_per_prompt),
        "--gpu-count", str(args.gpu_count),
        "--model-name", args.model_name,
        "--lora-rank", str(args.lora_rank),
        "--learning-rate", str(args.learning_rate),
        "--grpo-beta", str(args.grpo_beta),
        "--seq-len", str(args.seq_len),
        "--max-completion-tokens", str(args.max_completion_tokens),
    ]
    featurize = [
        "uv", "run", "python", "-m", "math_loop.features",
        "--raw", f"{run_dir}/raw",
        "--out", f"{run_dir}/parquet",
    ]
    pieces = _remote_env_prefix() + [shlex.join(collect), shlex.join(featurize)]
    return ["bash", "-lc", " && ".join(pieces)]


# Repo paths excluded from the on-pod upload (secrets, caches, large outputs).
UPLOAD_EXCLUDES = [".git", "outputs", "data/math_loop", ".venv", "pod_id.txt",
                   "private_key.pem", "public_key.pem"]


def rsync_upload_command(ssh: list[str], destination: str, repo_root: Path) -> list[str]:
    cmd = ["rsync", "-az", "--delete"]
    for pattern in UPLOAD_EXCLUDES:
        cmd += ["--exclude", pattern]
    cmd += ["-e", shlex.join(ssh), f"{repo_root}/", f"{destination}:{REMOTE_REPO}/"]
    return cmd


def tar_upload_commands(ssh: list[str], destination: str, repo_root: Path) -> tuple[list[str], list[str]]:
    """rsync-free upload: local `tar c` piped into a remote `tar x` over ssh."""
    local = ["tar", "czf", "-"]
    for pattern in UPLOAD_EXCLUDES:
        local += ["--exclude", pattern]
    local += ["-C", str(repo_root), "."]
    remote_cmd = [*ssh, destination,
                  f"mkdir -p {shlex.quote(REMOTE_REPO)} && tar xzf - -C {shlex.quote(REMOTE_REPO)}"]
    return local, remote_cmd


def tar_download_commands(ssh: list[str], destination: str, output_dir: Path) -> tuple[list[str], list[str]]:
    remote_cmd = [*ssh, destination,
                  f"tar czf - -C {shlex.quote(REMOTE_WORK + '/outputs')} ."]
    local = ["tar", "xzf", "-", "-C", str(output_dir)]
    return remote_cmd, local


def _run_pipe(producer: list[str], consumer: list[str], *, timeout: int = 3600) -> None:
    """Run `producer | consumer`, failing closed if either side errors."""
    p1 = subprocess.Popen(producer, stdout=subprocess.PIPE)
    assert p1.stdout is not None
    p2 = subprocess.Popen(consumer, stdin=p1.stdout)
    p1.stdout.close()  # allow p1 to get SIGPIPE if p2 exits
    rc2 = p2.wait(timeout=timeout)
    rc1 = p1.wait(timeout=timeout)
    if rc1:
        raise subprocess.CalledProcessError(rc1, producer)
    if rc2:
        raise subprocess.CalledProcessError(rc2, consumer)


def upload_repo(ssh: list[str], destination: str, repo_root: Path) -> None:
    if shutil.which("rsync"):
        run(rsync_upload_command(ssh, destination, repo_root))
        return
    producer, consumer = tar_upload_commands(ssh, destination, repo_root)
    print("[transport] rsync absent; uploading via tar-over-ssh", flush=True)
    _run_pipe(producer, consumer)


def download_outputs(ssh: list[str], destination: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        run(["rsync", "-az", "-e", shlex.join(ssh),
             f"{destination}:{REMOTE_WORK}/outputs/", f"{output_dir}/"])
        return
    producer, consumer = tar_download_commands(ssh, destination, output_dir)
    print("[transport] rsync absent; downloading via tar-over-ssh", flush=True)
    _run_pipe(producer, consumer)


def validate_smoke_outputs(output_dir: Path, args: argparse.Namespace) -> None:
    """After download, schema-validate the mini 4-Parquet and run tap.run_all."""
    parquet_dir = output_dir / "tap" / args.run_id / "parquet"
    if not parquet_dir.is_dir():
        raise RuntimeError(f"smoke produced no parquet dir at {parquet_dir}")
    repo_root = Path(__file__).resolve().parent
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"
    print(f"Validating mini-Parquet schema at {parquet_dir}...", flush=True)
    subprocess.run([sys.executable, "-m", "tap.schema", "--validate", str(parquet_dir)],
                   cwd=str(repo_root), env=env, check=True, timeout=600)
    report_out = output_dir / "tap" / args.run_id / "report"
    print("Running tap.run_all on the mini-Parquet...", flush=True)
    subprocess.run([sys.executable, "-m", "tap.run_all", "--parquet-dir", str(parquet_dir),
                    "--out", str(report_out)], cwd=str(repo_root), env=env, check=True, timeout=1200)


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

    print("Installing prime-rl (pinned) on the pod...", flush=True)
    remote(ssh, destination, build_bootstrap_command(args.prime_rl_commit),
           timeout=3600, log=log, stream=True)

    print("Running on-pod pre-flight (verifiers + AdvantageOutputs import)...", flush=True)
    remote(ssh, destination, build_preflight_command(), timeout=600, log=log, stream=True)

    print("Running TAP collection + featurization...", flush=True)
    remote(ssh, destination, build_collection_command(args),
           timeout=args.remote_timeout, log=log, stream=True)

    print("Downloading outputs...", flush=True)
    download_outputs(ssh, destination, output_dir)

    if args.smoke:
        validate_smoke_outputs(output_dir, args)


# ---------------------------------------------------------------------------
# Pod creation
# ---------------------------------------------------------------------------
def _pod_name(prefix: str) -> str:
    now = dt.datetime.now(dt.timezone.utc)
    return f"{prefix}{now:%Y%m%d-%H%M%S}"


def create_pod(args: argparse.Namespace) -> tuple[str, float | None]:
    """Create a pod, write pod_id.txt, and return (pod_id, price_per_hour)."""
    price_per_hour: float | None = None
    if args.offer_id:
        create = ["prime", "pods", "create", "--id", args.offer_id]
    else:
        resources = prime_json(
            "availability", "list",
            "--gpu-type", args.gpu_type,
            "--gpu-count", str(args.gpu_count),
            "--provider", args.provider,
            "--no-group-similar",
        )["gpu_resources"]
        offer = select_offer(resources, args)
        price_per_hour = offer.get("price_value")
        create = ["prime", "pods", "create", "--id", offer["id"]]
        for option, value in {
            "--disk-size": str(args.disk_size_gb or resource_value(offer["disk_gb"])),
            "--vcpus": resource_value(offer["vcpus"]),
            "--memory": resource_value(offer["memory_gb"]),
        }.items():
            create.extend((option, value))

    create.extend([
        "--name", _pod_name(args.pod_prefix),
        "--image", args.image,
        "--yes",
    ])
    pod_id = created_pod_id(run(create))
    _TeardownState.pod_id = pod_id
    POD_ID_FILE.write_text(pod_id + "\n", encoding="utf-8")
    return pod_id, price_per_hour


# ---------------------------------------------------------------------------
# SSH key handling
# ---------------------------------------------------------------------------
def prepare_ssh_key(key: Path) -> None:
    """chmod 600 the private key (ssh refuses 0644) and verify it matches the
    registered public key. Never generates a key."""
    if not key.is_file():
        raise SystemExit(f"SSH private key does not exist: {key}")
    try:
        key.chmod(0o600)
    except Exception as exc:
        raise SystemExit(f"could not chmod 600 {key}: {exc}") from None
    if DEFAULT_PUBLIC_KEY.is_file():
        try:
            derived = run(["ssh-keygen", "-y", "-f", str(key)], timeout=30).strip().split()
            expected = DEFAULT_PUBLIC_KEY.read_text(encoding="utf-8").strip().split()
            # Compare the key-type + base64 body (ignore the trailing comment).
            if derived[:2] != expected[:2]:
                raise SystemExit(
                    f"private key {key} does not match public key {DEFAULT_PUBLIC_KEY}; "
                    "refusing to launch with a mismatched/generated key."
                )
        except FileNotFoundError:
            print("[ssh] ssh-keygen not found; skipping pubkey match check", file=sys.stderr)


# ---------------------------------------------------------------------------
# Argument parsing + validation
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--cloud-id")
    parser.add_argument("--offer-id", help="skip offer selection and create this exact Prime offer id")
    parser.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE)
    parser.add_argument("--gpu-count", type=int, default=2)
    parser.add_argument("--image", default=PRIME_RL_IMAGE)
    parser.add_argument("--prime-rl-commit", default=None,
                        help="REQUIRED for a real launch: pin prime-rl to this commit sha.")
    parser.add_argument("--ssh-key", type=Path, default=DEFAULT_SSH_KEY)
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--max-price-per-hour", type=float)
    parser.add_argument("--max-cost-usd", type=float, default=DEFAULT_MAX_COST_USD)
    parser.add_argument("--max-wallclock-seconds", type=float, default=None)
    parser.add_argument("--disk-size-gb", type=int, default=1500)
    parser.add_argument("--pod-prefix", default=SMOKE_POD_PREFIX)
    parser.add_argument("--run-id", default="tap_smoke")
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--states", type=int, default=1, help="states per chain")
    parser.add_argument("--candidates-per-state", type=int, default=2)
    parser.add_argument("--prompts-per-candidate", type=int, default=2)
    parser.add_argument("--completions-per-prompt", type=int, default=4)
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--grpo-beta", type=float, default=0.04)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--max-completion-tokens", type=int, default=192)
    # Downloaded artifacts land at <output-dir>/tap/<run-id>/parquet, so the
    # default of "outputs" makes the canonical path outputs/tap/<run-id>/parquet.
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--runbook", type=Path, default=DEFAULT_RUNBOOK)
    parser.add_argument("--smoke", action="store_true",
                        help="cheap 1xH100 smoke: 1 state x 2 candidates, forces --gpu-count 1.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--remote-timeout", type=int, default=6 * 3600)
    # --keep-pod is deliberately NOT defined: this wave forbids keeping pods.
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    full_run_allowed = os.environ.get("TAP_ALLOW_FULL_RUN") == "1"

    if args.smoke:
        # The sanctioned cheap test: 2 chains x 1 state x 2 candidates. Two chains
        # keep tap.run_all's leave-one-chain-out split non-degenerate; prime-rl
        # needs >=2 GPUs (1 trainer + 1 inference), so the smoke runs on 2xH100.
        args.chains = 2
        args.states = 1
        args.candidates_per_state = 2
        args.gpu_count = max(args.gpu_count, 2)
        if args.gpu_count > 4 and not full_run_allowed:
            raise SystemExit("--smoke --gpu-count is capped at 4")
    else:
        # A non-smoke (full / collection) run requires the explicit env gate so
        # the loop can never launch the expensive 4xH100 collection on its own.
        if not full_run_allowed:
            raise SystemExit(
                "Refusing a non-smoke run: set TAP_ALLOW_FULL_RUN=1 to launch a real "
                "collection (the loop never sets it). Use --smoke for the cheap 2xH100 test."
            )

    if args.gpu_count < 2:
        raise SystemExit("--gpu-count must be at least 2 (prime-rl needs 1 trainer + 1 inference GPU)")

    if not args.pod_prefix.startswith(WAVE_POD_PREFIX):
        raise SystemExit(f"--pod-prefix must start with {WAVE_POD_PREFIX!r} so reaping is safe.")

    if args.max_cost_usd <= 0 or args.max_cost_usd > DEFAULT_MAX_COST_USD:
        raise SystemExit(f"--max-cost-usd must be in (0, {DEFAULT_MAX_COST_USD:.0f}].")

    if args.max_price_per_hour is not None and args.max_price_per_hour <= 0:
        raise SystemExit("--max-price-per-hour must be positive")

    if args.max_wallclock_seconds is None:
        args.max_wallclock_seconds = SMOKE_WALLCLOCK_SECONDS if args.smoke else FULL_WALLCLOCK_SECONDS

    # --prime-rl-commit is required for any real launch.
    if not args.dry_run and not args.prime_rl_commit:
        raise SystemExit("--prime-rl-commit <sha> is REQUIRED for a real launch (pin prime-rl).")


def print_plan(args: argparse.Namespace, repo_root: Path) -> None:
    commit = args.prime_rl_commit or resolve_prime_rl_head() or "<REQUIRED: pass --prime-rl-commit <sha>>"
    wall = args.max_wallclock_seconds or (SMOKE_WALLCLOCK_SECONDS if args.smoke else FULL_WALLCLOCK_SECONDS)
    print("Dry run. No pod will be created.")
    print("=== TAP v1 launch plan ===")
    print(f"provider:           {args.provider}")
    print(f"gpu:                {args.gpu_count} x {args.gpu_type}")
    print(f"image:              {args.image} (prime-rl base; prime-rl cloned + uv-synced on pod)")
    print(f"prime-rl commit pin: {commit}")
    print(f"cost cap:           ${args.max_cost_usd:.2f}  (fail-closed monitor: price/h x elapsed)")
    print(f"wall-clock cap:     {wall/3600:.2f}h  (independent absolute deadline)")
    print(f"pod name prefix:    {args.pod_prefix}")
    print(f"ssh key:            {args.ssh_key}")
    print(f"mode:               {'SMOKE (2 chains x 1 state x 2 candidates, 2xH100)' if args.smoke else 'FULL (requires TAP_ALLOW_FULL_RUN=1)'}")
    print("teardown:           atexit + SIGINT/SIGTERM + monitor breach ->")
    print(f"                    reap_pods.py --terminate-prefix {args.pod_prefix} AND --terminate-id <pod_id.txt>")
    print(f"                    (--keep-pod is forbidden; pod_id.txt written on create)")
    print(f"collection:         {args.chains} chains x {args.states} states x "
          f"{args.candidates_per_state} candidates -> "
          f"{args.chains * args.states * args.candidates_per_state} labels")
    print(f"would upload:       {repo_root} -> {REMOTE_REPO}")
    print("bootstrap:")
    print("  " + shlex.join(build_bootstrap_command(commit)))
    print("pre-flight (verifiers + AdvantageOutputs import):")
    print("  " + shlex.join(build_preflight_command()))
    print("collection + featurize:")
    print("  " + shlex.join(build_collection_command(args)))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    args.ssh_key = args.ssh_key.expanduser().resolve()
    validate_args(args)

    _TeardownState.prefix = args.pod_prefix
    _TeardownState.runbook = args.runbook

    if args.dry_run:
        print_plan(args, repo_root)
        return

    prepare_ssh_key(args.ssh_key)

    install_safety_handlers()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pod_id, price_per_hour = create_pod(args)
    monitor = CostWallclockMonitor(
        price_per_hour=price_per_hour,
        max_cost_usd=args.max_cost_usd,
        max_wallclock_seconds=args.max_wallclock_seconds,
        runbook=args.runbook,
    )
    monitor.start()
    try:
        run_on_pod(args, pod_id, repo_root, output_dir)
    finally:
        monitor.stop()
        print(f"Tearing down pod {pod_id}", flush=True)
        teardown(reason="run complete")
    print(f"Outputs saved under {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        teardown(reason="subprocess failure")
        raise SystemExit(exc.returncode) from None
