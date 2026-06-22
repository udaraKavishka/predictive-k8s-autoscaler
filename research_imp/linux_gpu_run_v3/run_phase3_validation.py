#!/usr/bin/env python
"""
Phase 3 of 3 - Day 3 (~9-10 h): naive-FT + cross-app + SLA + dashboard.

Everything up to `forgetting` loads from cache. This phase computes:
  - naive_ft (~9 h): plain sequential fine-tuning, the no-CL BWT reference. It
    compiles the optimizer once (not per chunk) and clears the GPU graph on entry,
    so it does not OOM the way the original crashed run did.
  - cross_app generalisation, the SLA summary, and the dashboard.

Run on Day 3, after Phase 2 finished. This is the last phase.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "research_imp_V3_linux_gpu.py")

for name in ["naive_ft_results.json", "cross_app_results.json"]:
    path = os.path.join(HERE, name)
    if os.path.exists(path):
        os.remove(path)
        print(f"removed stale {name}")

cmd = [
    sys.executable,
    SCRIPT,
    "--run-until", "dashboard",
    "--skip-multi-seed",
    "--skip-ablation",
    "--skip-sensitivity",
]

print("=== Phase 3/3: naive_ft -> dashboard (--run-until dashboard) ===")
print("Running:", " ".join(cmd))
raise SystemExit(subprocess.call(cmd, cwd=HERE))
