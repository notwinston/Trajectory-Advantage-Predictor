"""Wave-2 hardening tests for run_prime_rl_math_loop.py and reap_pods.py.

These exercise the safety rails with NO Prime Intellect calls:
  * SIGTERM fires the teardown handler (reap-by-prefix + terminate-by-id),
  * reap_pods refuses empty / non-`tap-v1-` prefixes,
  * --keep-pod is forbidden, --gpu-count>1 needs the TAP_ALLOW_FULL_RUN gate,
  * a real launch requires --prime-rl-commit, --smoke forces gpu=1,
  * the dry-run --smoke plan advertises lambdalabs + pinned commit + $80 cap +
    teardown.
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import reap_pods  # noqa: E402
import run_prime_rl_math_loop as launcher  # noqa: E402


class SigtermTeardownTest(unittest.TestCase):
    def test_sigterm_fires_teardown(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "marker.txt"
            child = textwrap.dedent(
                f"""
                import os, signal, time
                import run_prime_rl_math_loop as L
                marker = {str(marker)!r}
                def fake_reap(prefix):
                    with open(marker, "a") as fh: fh.write("reap:" + prefix + "\\n")
                def fake_term(pod_id):
                    with open(marker, "a") as fh: fh.write("term:" + pod_id + "\\n")
                L._reap_prefix = fake_reap
                L._terminate_id = fake_term
                L._TeardownState.prefix = "tap-v1-smoke-"
                L._TeardownState.pod_id = "pod-xyz"
                L.install_safety_handlers()
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(5)
                os._exit(99)  # should never run; handler exits first
                """
            )
            result = subprocess.run(
                [sys.executable, "-c", child],
                cwd=str(REPO_ROOT), text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(result.returncode, launcher.EXIT_SIGNAL, result.stderr)
            contents = marker.read_text(encoding="utf-8")
            self.assertIn("reap:tap-v1-smoke-", contents)
            self.assertIn("term:pod-xyz", contents)

    def test_teardown_is_idempotent(self):
        calls = []
        with mock.patch.object(launcher, "_reap_prefix", lambda p: calls.append(("reap", p))), \
             mock.patch.object(launcher, "_terminate_id", lambda i: calls.append(("term", i))):
            launcher._TeardownState.done = False
            launcher._TeardownState.prefix = "tap-v1-smoke-"
            launcher._TeardownState.pod_id = "pod-1"
            launcher.teardown()
            launcher.teardown()  # second call must be a no-op
        self.assertEqual(calls.count(("reap", "tap-v1-smoke-")), 1)
        self.assertEqual(calls.count(("term", "pod-1")), 1)
        launcher._TeardownState.done = False  # reset for other tests


class ReapPrefixGuardTest(unittest.TestCase):
    def test_refuses_empty_prefix(self):
        self.assertEqual(reap_pods.cmd_terminate_prefix(""), 2)

    def test_refuses_non_wave_prefix(self):
        self.assertEqual(reap_pods.cmd_terminate_prefix("prod-"), 2)
        self.assertEqual(reap_pods.cmd_terminate_prefix("smoke-"), 2)

    def test_refuses_empty_id(self):
        self.assertEqual(reap_pods.cmd_terminate_id(""), 2)

    def test_accepts_wave_prefix_without_calling_prime_when_no_pods(self):
        with mock.patch.object(reap_pods, "list_pods", lambda: []):
            self.assertEqual(reap_pods.cmd_terminate_prefix("tap-v1-smoke-"), 0)


class ArgGateTest(unittest.TestCase):
    def test_keep_pod_is_forbidden(self):
        with self.assertRaises(SystemExit):
            launcher.parse_args(["--keep-pod"])

    def test_real_run_requires_commit(self):
        # --smoke is allowed unattended, but a real launch still needs a pin.
        args = launcher.parse_args(["--smoke"])
        with self.assertRaises(SystemExit) as ctx:
            launcher.validate_args(args)
        self.assertIn("prime-rl-commit", str(ctx.exception))

    def test_non_smoke_requires_full_run_gate(self):
        args = launcher.parse_args(["--prime-rl-commit", "abc123"])
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TAP_ALLOW_FULL_RUN", None)
            with self.assertRaises(SystemExit) as ctx:
                launcher.validate_args(args)
        self.assertIn("TAP_ALLOW_FULL_RUN", str(ctx.exception))

    def test_smoke_forces_gpu_count_1(self):
        args = launcher.parse_args(["--smoke", "--gpu-count", "4", "--dry-run",
                                    "--prime-rl-commit", "abc123"])
        launcher.validate_args(args)
        self.assertEqual(args.gpu_count, 1)
        self.assertEqual(args.states, 1)
        self.assertEqual(args.candidates_per_state, 2)

    def test_full_run_gate_allows_4gpu_with_commit(self):
        args = launcher.parse_args(["--gpu-count", "4", "--prime-rl-commit", "abc123",
                                    "--dry-run"])
        with mock.patch.dict(os.environ, {"TAP_ALLOW_FULL_RUN": "1"}):
            launcher.validate_args(args)  # must not raise
        self.assertEqual(args.gpu_count, 4)

    def test_full_run_gate_still_requires_commit(self):
        args = launcher.parse_args(["--gpu-count", "4"])
        with mock.patch.dict(os.environ, {"TAP_ALLOW_FULL_RUN": "1"}):
            with self.assertRaises(SystemExit) as ctx:
                launcher.validate_args(args)
        self.assertIn("prime-rl-commit", str(ctx.exception))

    def test_pod_prefix_must_be_wave_guarded(self):
        args = launcher.parse_args(["--smoke", "--dry-run", "--prime-rl-commit", "abc",
                                    "--pod-prefix", "evil-"])
        with self.assertRaises(SystemExit) as ctx:
            launcher.validate_args(args)
        self.assertIn("tap-v1-", str(ctx.exception))

    def test_max_cost_capped_at_80(self):
        args = launcher.parse_args(["--smoke", "--dry-run", "--prime-rl-commit", "abc",
                                    "--max-cost-usd", "200"])
        with self.assertRaises(SystemExit):
            launcher.validate_args(args)


class TransportTest(unittest.TestCase):
    SSH = ["ssh", "-i", "/workspace/private_key.pem", "-p", "22"]

    def test_rsync_upload_excludes_secrets(self):
        cmd = launcher.rsync_upload_command(self.SSH, "user@host", Path("/workspace"))
        self.assertEqual(cmd[0], "rsync")
        for secret in ("private_key.pem", "public_key.pem", ".git", ".venv", "outputs"):
            self.assertIn(secret, cmd)

    def test_tar_upload_pipes_and_excludes_secrets(self):
        producer, consumer = launcher.tar_upload_commands(self.SSH, "user@host", Path("/workspace"))
        self.assertEqual(producer[:3], ["tar", "czf", "-"])
        self.assertIn("private_key.pem", producer)
        self.assertIn("public_key.pem", producer)
        self.assertEqual(consumer[0], "ssh")
        self.assertIn("user@host", consumer)
        self.assertIn("tar xzf - -C", consumer[-1])

    def test_tar_download_pulls_remote_outputs(self):
        remote_cmd, local = launcher.tar_download_commands(self.SSH, "user@host", Path("/tmp/out"))
        self.assertEqual(remote_cmd[0], "ssh")
        self.assertIn("tar czf -", remote_cmd[-1])
        self.assertEqual(local[:3], ["tar", "xzf", "-"])
        self.assertEqual(local[-1], "/tmp/out")


class DryRunPlanTest(unittest.TestCase):
    def test_dry_run_smoke_plan_advertises_safety(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            launcher.main(["--dry-run", "--smoke", "--prime-rl-commit", "deadbeefcafe"])
        out = buf.getvalue()
        self.assertIn("lambdalabs", out)
        self.assertIn("deadbeefcafe", out)
        self.assertIn("$80", out)
        self.assertIn("tap-v1-smoke-", out)
        self.assertIn("reap_pods.py", out)
        self.assertIn("forbidden", out)


if __name__ == "__main__":
    unittest.main()
