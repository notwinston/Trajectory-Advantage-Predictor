#!/usr/bin/env python3
"""One monitoring pass over the 4 TAP domains: alive? progressing?
Prints a STATUS line every run; emits an ALERT/DONE sentinel only on a real
event (crash, stall, or completion) so the loop only wakes the agent then."""
import json, os, re, subprocess

STATE = "/tmp/tap_domain_monitor.json"
KEY = os.path.expanduser("~/.ssh/id_ed25519")
SSH = ["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=no", "-o",
       "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
COUNT = ('find /workspace/math_loop_runs/outputs -name "labels_shard_*.jsonl" '
         '! -name "*rollouts*" -exec cat {} + 2>/dev/null | wc -l')
TARGET = 240            # 80 cohorts * 3 seeds == "done"
STALL = 4               # producing then flat this many checks => stall
STUCK0 = 12             # ACTIVE but 0 labels this many checks => stuck (all MCQ now, 1st cohort is fast)

# domain -> candidate pod ids (science keeps new+old during the handoff)
DOMAINS = {  # all 4 MCQ collection done -> now tracking strong-treatment validations
    "val_sci":  ["e6fa9020251442dfadd6e8b7b758adbb"],
    "val_math": ["8949bad2d5374d1ebd2fc2b7afb9670d"],
}

def pstatus(pid):
    try:
        r = subprocess.run(["prime", "pods", "status", pid, "--output", "json"],
                           capture_output=True, text=True, timeout=30)
        d = json.loads(r.stdout)
        st = str(d.get("status", "?")).upper()
        ssh = d.get("ssh", "")
        ssh = ssh[0] if isinstance(ssh, list) and ssh else (ssh if isinstance(ssh, str) else "")
        m = re.search(r"(\w+)@([0-9.]+)", str(ssh))
        return st, (m.group(1) if m else None), (m.group(2) if m else None)
    except Exception:
        return "ERR", None, None

def count(user, ip):
    if not ip:
        return -1
    try:
        r = subprocess.run(SSH + [f"{user}@{ip}", COUNT], capture_output=True, text=True, timeout=25)
        s = (r.stdout.strip() or "").split()
        return int(s[-1]) if s and s[-1].isdigit() else -1
    except Exception:
        return -1

prev = {}
if os.path.exists(STATE):
    try:
        prev = json.load(open(STATE))
    except Exception:
        prev = {}

now, alerts, done, lines = {}, [], [], []
for dom, pids in DOMAINS.items():
    best_lab, best_st, alive = -1, "?", False
    for pid in pids:
        st, user, ip = pstatus(pid)
        if st in ("ACTIVE", "PROVISIONING"):
            alive = True
        lab = count(user, ip) if st == "ACTIVE" else (0 if st == "PROVISIONING" else -1)
        if lab > best_lab:
            best_lab, best_st = lab, st
    p = prev.get(dom, {})
    pl, flat = p.get("labels", -1), p.get("flat", 0)
    if not alive:
        alerts.append(f"{dom}=DOWN({best_st})"); flat = 0
    elif best_lab > pl:
        flat = 0
    else:
        flat += 1
        if best_lab > 0 and flat >= STALL:
            alerts.append(f"{dom}=STALL({best_lab} flat {flat}m)")
        elif best_lab == 0 and best_st == "ACTIVE" and flat >= STUCK0:
            alerts.append(f"{dom}=STUCK(active,0 labels,{flat}m)")
    if best_lab >= TARGET and not p.get("done"):
        done.append(f"{dom}={best_lab}")
    now[dom] = {"labels": best_lab, "flat": flat, "done": best_lab >= TARGET}
    d = (best_lab - pl) if pl >= 0 else 0
    lines.append(f"{dom}:{best_st[:4]} n={best_lab} +{d if d>0 else 0} flat={flat}")

json.dump(now, open(STATE, "w"))
print("STATUS " + " | ".join(lines), flush=True)
if alerts:
    print("AGENT_LOOP_ALERT_domains " + json.dumps(
        {"prompt": "Investigate TAP domain issue(s): " + ", ".join(alerts) +
         ". SSH the pod, tail the shard logs for errors, relaunch the domain if it crashed.",
         "alerts": alerts}), flush=True)
if done:
    print("AGENT_LOOP_DONE_domains " + json.dumps(
        {"prompt": "Domain(s) hit the 240-record target: " + ", ".join(done) +
         ". Download labels and terminate that pod to save cost.", "done": done}), flush=True)
