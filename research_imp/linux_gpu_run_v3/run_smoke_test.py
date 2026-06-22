#!/usr/bin/env python
"""
OPTIONAL smoke test (~1 h): run the WHOLE pipeline end-to-end with epochs=1.

Purpose: validate that the code actually works under the current TF/Keras-3 stack
(train -> save .h5 -> reload -> baselines -> naive_ft -> cross_app -> dashboard)
BEFORE committing to the ~30 h real run. The numbers are meaningless at 1 epoch;
this only checks for API/compatibility/OOM failures.

Writes everything to a separate ./smoke/ subdir so it does not touch the real run.
Run this first; if it finishes without error, start run_phase1_train.py.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "research_imp_V3_linux_gpu.py")
CSV = os.path.join(HERE, "alibaba_timeseries_full.csv")
SMOKE_DIR = os.path.join(HERE, "smoke")
os.makedirs(SMOKE_DIR, exist_ok=True)

cmd = [
    sys.executable,
    SCRIPT,
    "--preprocessed-path", CSV,
    "--output-dir", SMOKE_DIR,
    "--epochs", "1",
    "--run-until", "dashboard",
    "--skip-multi-seed",
    "--skip-ablation",
    "--skip-sensitivity",
]

print("=== SMOKE TEST: full pipeline, epochs=1, output -> ./smoke/ ===")
print("Running:", " ".join(cmd))
raise SystemExit(subprocess.call(cmd, cwd=HERE))
