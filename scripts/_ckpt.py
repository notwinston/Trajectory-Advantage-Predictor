#!/usr/bin/env python3
"""Checkpoint every live qwen3 pod's labels to local outputs/ckpt/<pid>/ so
in-flight labels are NEVER pod-only (a dead pod can't lose them again)."""
import json, os, re, subprocess

KEY = os.path.expanduser("~/.ssh/id_ed25519")
SSHOPT = ("-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
          "-o ConnectTimeout=12 -o BatchMode=yes")

def live_pods():
    r = subprocess.run(["prime", "pods", "list", "--output", "json"],
                       capture_output=True, text=True, timeout=40)
    d = json.loads(r.stdout)
    ps = d.get("pods", d) if isinstance(d, dict) else d
    return [p for p in ps if "qwen3" in str(p.get("name", "")).lower()
            and str(p.get("status", "")).upper() == "ACTIVE"]

def ip_user(pid):
    r = subprocess.run(["prime", "pods", "status", pid, "--output", "json"],
                       capture_output=True, text=True, timeout=30)
    d = json.loads(r.stdout)
    ssh = d.get("ssh", "")
    ssh = ssh[0] if isinstance(ssh, list) and ssh else ssh
    m = re.search(r"(\w+)@([0-9.]+)", str(ssh))
    return (m.group(1), m.group(2)) if m else (None, None)

saved = 0
for p in live_pods():
    pid = p.get("id")
    user, ip = ip_user(pid)
    if not ip:
        continue
    dst = f"outputs/ckpt/{pid[:8]}/"
    os.makedirs(dst, exist_ok=True)
    cmd = ["rsync", "-az", "-e", f"ssh -i {KEY} {SSHOPT}",
           f"{user}@{ip}:/workspace/math_loop_runs/outputs/", dst]
    try:
        subprocess.run(cmd, timeout=120, check=False)
        n = subprocess.run("cat %s/labels_shard_*.jsonl 2>/dev/null | grep -vc rollouts" % dst,
                           shell=True, capture_output=True, text=True).stdout.strip()
        print(f"  ckpt {pid[:8]} {user}@{ip} -> {dst} (~{n} labels)", flush=True)
        saved += 1
    except Exception as e:
        print(f"  ckpt {pid[:8]} FAILED: {e}", flush=True)
print(f"CKPT_DONE ({saved} pods)", flush=True)
