#!/usr/bin/env python
"""
Phase 2 of 3 - Day 2 (~10-11 h): baselines + core outputs + summary + forgetting.

With the proposed model from Phase 1 cached, this phase trains the baselines
(Static LSTM, Static Hybrid, and Periodic) from scratch, then computes avg_task
metrics, the core tables/figures, the summary, and the forgetting (BWT) deep dive.

NOTE: because this is a fresh Keras-3 run with no reusable .h5 models, the Periodic
baseline trains over all 4 chunks (~9 h), so this phase is ~10-11 h, not ~3 h.

Run on Day 2, after Phase 1 finished. Then run run_phase3_validation.py on Day 3.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "research_imp_V3_linux_gpu.py")

# Remove any stale outputs from a previous attempt so they recompute cleanly.
for name in ["forgetting_results.json", "avg_task_metrics.csv"]:
    path = os.path.join(HERE, name)
    if os.path.exists(path):
        os.remove(path)
        print(f"removed stale {name}")

cmd = [
    sys.executable,
    SCRIPT,
    "--run-until", "forgetting",
    "--skip-multi-seed",
    "--skip-ablation",
    "--skip-sensitivity",
]

print("=== Phase 2/3: baselines -> forgetting (--run-until forgetting) ===")
print("Running:", " ".join(cmd))
raise SystemExit(subprocess.call(cmd, cwd=HERE))
