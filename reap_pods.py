#!/usr/bin/env python3
"""Reap Prime Intellect pods for TAP v1 by id or by name prefix.

Safety: --terminate-prefix REFUSES an empty prefix or any prefix that does not
start with the wave guard `tap-v1-`, so a stray invocation can never terminate
unrelated pods on the account. `--list` prints the active `tap-v1-` pods (and a
final count) so the launcher can confirm 0 pods remain after teardown.

Uses the `prime` CLI; PRIME_API_KEY must be in the environment (never written to
disk). torch/transformers are never imported.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


WAVE_POD_PREFIX = "tap-v1-"
ACTIVE_STATES = {"ACTIVE", "PROVISIONING", "PENDING", "INSTALLING", "STARTING", "RUNNING"}


def _run(command: list[str], *, timeout: int = 300) -> str:
    result = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=timeout, check=False,
    )
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command, result.stdout)
    return result.stdout


def _coerce_pod_list(payload) -> list[dict]:
    """Normalize the several shapes `prime pods list --output json` may return."""
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        for key in ("pods", "data", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [p for p in value if isinstance(p, dict)]
        # single pod object
        if payload.get("id") or payload.get("pod_id"):
            return [payload]
    return []


def list_pods() -> list[dict]:
    out = _run(["prime", "pods", "list", "--output", "json"])
    return _coerce_pod_list(json.loads(out))


def _pod_id(pod: dict) -> str:
    return str(pod.get("id") or pod.get("pod_id") or pod.get("uuid") or "")


def _pod_name(pod: dict) -> str:
    return str(pod.get("name") or pod.get("pod_name") or "")


def _pod_status(pod: dict) -> str:
    return str(pod.get("status") or pod.get("state") or "").upper()


def _is_active(pod: dict) -> bool:
    status = _pod_status(pod)
    if not status:
        return True  # unknown status -> treat as active to be safe
    return status not in {"TERMINATED", "CANCELLED", "ERROR", "FAILED", "STOPPED", "DELETED"}


def terminate(pod_id: str) -> None:
    if not pod_id:
        return
    print(f"terminating pod {pod_id}", flush=True)
    _run(["prime", "pods", "terminate", pod_id, "--yes"])


def cmd_list(only_wave: bool = True) -> int:
    pods = [p for p in list_pods() if _is_active(p)]
    wave = [p for p in pods if _pod_name(p).startswith(WAVE_POD_PREFIX)]
    shown = wave if only_wave else pods
    for pod in shown:
        print(f"{_pod_id(pod)}\t{_pod_name(pod)}\t{_pod_status(pod)}")
    label = "tap-v1-" if only_wave else "all"
    print(f"ACTIVE {label} pods: {len(wave) if only_wave else len(pods)}")
    return 0


def cmd_terminate_prefix(prefix: str) -> int:
    if not prefix or not prefix.startswith(WAVE_POD_PREFIX):
        print(
            f"refusing to terminate-by-prefix {prefix!r}: prefix must be non-empty and "
            f"start with {WAVE_POD_PREFIX!r}",
            file=sys.stderr,
        )
        return 2
    pods = [p for p in list_pods() if _is_active(p) and _pod_name(p).startswith(prefix)]
    if not pods:
        print(f"no active pods match prefix {prefix!r}")
        return 0
    failures = 0
    for pod in pods:
        try:
            terminate(_pod_id(pod))
        except Exception as exc:  # keep reaping the rest
            print(f"failed to terminate {_pod_id(pod)}: {exc!r}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


def cmd_terminate_id(pod_id: str) -> int:
    if not pod_id:
        print("refusing to terminate empty pod id", file=sys.stderr)
        return 2
    terminate(pod_id)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="list active tap-v1- pods + count")
    group.add_argument("--terminate-prefix", metavar="PREFIX",
                       help="terminate active pods whose name starts with PREFIX (must start with tap-v1-)")
    group.add_argument("--terminate-id", metavar="POD_ID", help="terminate a single pod by id")
    parser.add_argument("--all", action="store_true", help="with --list, show all pods not just tap-v1-")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        return cmd_list(only_wave=not args.all)
    if args.terminate_prefix is not None:
        return cmd_terminate_prefix(args.terminate_prefix)
    if args.terminate_id is not None:
        return cmd_terminate_id(args.terminate_id)
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(exc.output or "", file=sys.stderr)
        raise SystemExit(exc.returncode) from None
