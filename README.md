# Adaptive CPU Auto-Scaling Using Hybrid Neural Networks and Continual Learning in Kubernetes Orchestration

Predictive, continual-learning driven CPU autoscaling for Kubernetes CI/CD workloads — hybrid LSTM+MLP forecasting with EWC + Experience Replay to adapt to non‑stationary workloads while preserving historical knowledge.

## Overview

This repository contains research code and reproducible experiments for an adaptive CPU autoscaling framework that combines:

- A hybrid neural model (LSTM branch for temporal features + MLP branch for static features)
- Continual learning techniques: Elastic Weight Consolidation (EWC) and Experience Replay (ER)
- A simulation and evaluation harness (workload replay, Prometheus/Grafana integration, metrics)

The approach targets DevOps/CI‑CD workloads (bursty, job-driven) and aims to reduce over‑provisioning, improve SLA compliance, and retain long‑term performance as workloads evolve.

## Key Features

- Hybrid forecasting model: separate temporal and static feature branches with fusion layer
- Continual learning engine: Fisher-based regularization (EWC) + replay buffer (ER)
- Controller prototype that converts forecasts to safe Kubernetes scaling actions with HPA fallback
- Evaluation scripts, preprocessing pipelines, and benchmarks using Alibaba cluster traces or user datasets

## Repository structure

- `Notebooks/` — analysis and experiments
- `research_imp/` — implementation (models, training, replay, EWC)
- `k8s_autoscaler/` — controller prototype, Kubernetes manifests and tooling
- `data/` — preprocessed datasets and large artifacts
- `models/`  — model checkpoints and weights
- `README.md` — this file
- `requirements.txt` — Python dependencies

## Quick start

1. Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Preprocess the trace data (example):

```bash
python Notebooks/build_timeseries_full.py --input /path/to/alibaba_trace --out data/preprocessed.npz
```

3. Train a baseline hybrid model:

```bash
python research_imp/train_hybrid.py --data data/preprocessed.npz --out models/hybrid_initial
```

<!-- 4. Run the simulator (local Kubernetes via `kind` or `minikube`):

```bash
# start local cluster and deploy controller/proxy
# follow instructions in k8s_autoscaler/README.md
``` -->

## Evaluation

Scripts in `research_imp/` compute MAE, RMSE, MAPE and operational metrics (SLA violations, cost, utilization). Reproducible experiments use fixed seeds and archived checkpoints.

## Data and ethics

Datasets (e.g., Alibaba traces) must be anonymized. Do not commit raw traces or PII.

## Citation

If you use this work, please cite the accompanying paper or the project repository once published.

## License

This project is released under the MIT License by default. To change the license, add a `LICENSE` file in the repository root.

## Contact

Primary author / maintainer: udara nalawansa <udarakavishka13@gmail.com>
