# Linux GPU run (V3) — fresh clean retrain

Self-contained folder for running the V3 pipeline on **Linux + NVIDIA GPU** under a
modern **TF 2.21 / Keras 3.14** stack (conda env `research_env`). It starts empty
(no old `.h5` models), so everything trains from scratch in the current environment.

## Why this folder exists
The previous `Latest run win_gpu v3/` artifacts were produced by **TF 2.10 / Keras 2**
on Windows. Keras 3 cannot load those `.h5` files (`Unrecognized keyword argument
'time_major'`). Rather than fight cross-version loading, this folder does a clean
retrain. The pipeline also now has:
- a **cache-compatibility guard** (retrains instead of crashing if a model can't load
  or has the wrong feature width), and
- the **naive_ft GPU-OOM fix** (optimizer compiled once, GPU graph cleared before the
  heavy validation training).

Static features are **6** (the leaky `instance_count` was removed); EWC `lambda=100`,
replay ratio `0.2`.

## Contents
- `research_imp_V3_linux_gpu.py` — the pipeline (with the fixes above)
- `alibaba_timeseries_full.csv` — the preprocessed dataset
- `run_smoke_test.py` — optional ~1 h end-to-end check (epochs=1) before the real run
- `run_phase1_train.py` / `run_phase2_baselines.py` / `run_phase3_validation.py`

## How to run
Activate the env first, e.g. `conda activate research_env`, then from this folder:

```
python run_smoke_test.py        # optional, ~1 h — verify nothing breaks under Keras 3
python run_phase1_train.py      # Day 1, ~10-11 h  (proposed model)
python run_phase2_baselines.py  # Day 2, ~10-11 h  (baselines + forgetting)
python run_phase3_validation.py # Day 3, ~9-10 h   (naive_ft + cross_app + dashboard)
```

Each phase resumes from the previous one via the on-disk manifest, so you can run them
on separate days. Total ≈ 30 h split across three days.

## If the smoke test fails
It means the training code hits a Keras-3 API incompatibility. Capture the traceback
and fix the specific call before spending the multi-day run.
