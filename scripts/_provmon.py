#!/usr/bin/env python3
"""Continuously watch pod provisioning; flag STUCK early, confirm ACTIVE."""
import json, subprocess, time, datetime, sys

PODS = {
    "math":    "deb8303950764738822442ed939a15fd",
    "code":    "f2d30fbfdcc8473384ccfa077e182f91",
    "mmlu":    "f4938cfeb37040e38a53b1b2011f15ae",
    "science": "be773e9dc2504f8f97e909ae01626f95",
}
STUCK_MIN = 8.0   # provisioning longer than this == stuck (lambda usually ACTIVE < 6m)

def age_min(created):
    try:
        t = datetime.datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        return (datetime.datetime.now(datetime.timezone.utc) - t).total_seconds() / 60
    except Exception:
        return -1.0

for n in range(80):
    print(f"===== {time.strftime('%H:%M:%S')} iter {n+1} =====", flush=True)
    all_active = True
    stuck = []
    for dom, pid in PODS.items():
        try:
            r = subprocess.run(["prime", "pods", "status", pid, "--output", "json"],
                               capture_output=True, text=True, timeout=30)
            d = json.loads(r.stdout)
        except Exception as e:
            print(f"  {dom} {pid[:8]}: STATUS_ERR ({e})", flush=True)
            all_active = False
            continue
        st = str(d.get("status", "?")).upper()
        am = age_min(d.get("created_at", ""))
        inst = str(d.get("installation_status", "?"))
        print(f"  {dom:8s} {pid[:8]}: {st:13s} age={am:5.1f}m install={inst}", flush=True)
        if st != "ACTIVE":
            all_active = False
            if "PROVISION" in st and am > STUCK_MIN:
                stuck.append(f"{dom}({am:.0f}m)")
        if st in ("TERMINATED", "ERROR", "FAILED", "CANCELLED"):
            stuck.append(f"{dom}=DEAD")
    if stuck:
        print("STUCK_OR_DEAD: " + ", ".join(stuck), flush=True)
    if all_active:
        print("ALL_ACTIVE", flush=True)
        break
    time.sleep(30)
print("MONITOR_DONE", flush=True)
