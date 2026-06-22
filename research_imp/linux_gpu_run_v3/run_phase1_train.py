#!/usr/bin/env python
"""
Phase 1 of 3 - Day 1 (~10-11 h): train the proposed EWC+ER model (6 features).

This folder starts with no cached .h5 models, so the proposed model is trained
from scratch in the current (Keras 3) environment. Stops after the `train` stage,
the long sequential continual-learning step over the 4 chunks.

Run on Day 1. When it finishes, run run_phase2_baselines.py on Day 2.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "research_imp_V3_linux_gpu.py")

cmd = [
    sys.executable,
    SCRIPT,
    "--run-until", "train",
    "--skip-multi-seed",
    "--skip-ablation",
    "--skip-sensitivity",
]

print("=== Phase 1/3: train proposed model (--run-until train) ===")
print("Running:", " ".join(cmd))
raise SystemExit(subprocess.call(cmd, cwd=HERE))
