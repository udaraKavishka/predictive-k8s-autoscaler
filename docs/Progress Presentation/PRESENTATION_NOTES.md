# Research Progress Presentation Notes
## Adaptive CPU Auto-Scaling Using Hybrid Neural Networks and Continual Learning in Kubernetes

> **Format:** 5 minutes presentation · 7 minutes Q&A  
> **Timing guide:** ~1 minute per section heading below

---

Each presentation will consist of 5 minutes of presentation time, followed by 7 minutes allocated for questions and discussion. Please ensure your presentation clearly addresses the following:

* Research objectives and scope
* Progress to date
* Remaining work
* Planned vs. actual progress

---

## Slide 1 — Title & One-Line Summary (30 sec)

**Title:** Adaptive CPU Auto-Scaling Using Hybrid Neural Networks and Continual Learning in Kubernetes Orchestration

**One-liner:** A framework that predicts CPU demand 30 minutes ahead using a hybrid LSTM+MLP model that continuously adapts to evolving workloads without forgetting historical patterns — replacing reactive threshold-based Kubernetes scaling.

---

## Slide 2 — Research Objectives & Scope (1 min)

### Problem being solved
Standard Kubernetes Horizontal Pod Autoscaler (HPA) reacts *after* CPU thresholds are breached — too late to prevent performance degradation. Static ML models trained on past data degrade as workloads evolve, and full retraining causes catastrophic forgetting of earlier patterns.

### Core objective
Design and evaluate a CPU forecasting and auto-scaling system that:
- Predicts demand **30 minutes ahead** (6 × 5-min intervals)
- Uses a **hybrid two-branch neural network** (LSTM for temporal sequences + MLP for static job features)
- Applies **continual learning** (Elastic Weight Consolidation + Experience Replay) to adapt to new patterns without forgetting old ones
- Targets **Kubernetes-based CI/CD pipelines** running on Alibaba cluster workloads

### Scope boundaries
- Dataset: Alibaba Cluster Traces (large-scale containerised cloud production data)
- Simulation environment: Minikube/kind cluster with Prometheus + Grafana monitoring
- Comparison: 4 baselines — Kubernetes HPA, Static LSTM, Static Hybrid model, Periodic Retraining
- Evaluation: MAE, RMSE, SLA violation rate, Backward Transfer (catastrophic forgetting metric)

---

## Slide 3 — Progress to Date (1.5 min)

### Completed
| Component | Status |
|---|---|
| Literature review (18 papers, 5 research gaps identified) | Done |
| Dataset acquired and preprocessed (Alibaba traces, ~1M+ sequences) | Done |
| Feature engineering (10 temporal + 7 static features per sample) | Done |
| Hybrid LSTM+MLP architecture implemented and trained | Done |
| EWC + Experience Replay continual learning engine | Done |
| All 4 baseline models trained (HPA-reactive, Static LSTM, Static Hybrid, Periodic Retrain) | Done |
| Core evaluation figures generated (`evaluation_results.png`, `validation_dashboard.png`) | Done |
| SLA analysis exported | Done |
| BWT (Backward Transfer) scalar computed | Done |

### Key results — V2 run (current)
- Proposed model **outperforms ALL neural baselines** including Periodic Retrain (MAE: 426 vs 439) ✓
- Proposed model **SLA violations: 0.30%** — near-identical to reactive HPA (0.25%); 20× better than Periodic Retrain (7.43%) ✓
- MAE decreases progressively C1→C2→C3 (1609→964→302); C4 slight rise to 426 (concept drift) ✓
- RMSE/MAE ratio = 1.03 — consistent predictions, no dangerous spikes ✓
- BWT = −240: C1 improved 53% after all chunks (positive backward transfer); C3 forgetting reduced from 39% (V1) to 15% (V2) ✓
- Under-provisioning (UnderProv%) = 1.42% — lowest of all neural methods, explaining the near-zero SLA violations ✓

### V2 fixes confirmed working
- HPA baseline: corrected to 6-step horizon (MAE=1.26, up from bug value of 0.28) ✓
- EWC lambda: increased 100→500; replay buffer: 500→2000; epochs: 10→30 ✓
- MAPE filtered to active demand (> p10); OverProv/UnderProv normalised to % of total demand ✓

### Remaining pending (not yet run)
- Ablation study (6 variants), multi-seed CI, sensitivity sweeps, cross-app evaluation — all implemented in V2, need `--force-validation` run

---

## Slide 4 — Remaining Work (1 min)

| Task | Priority | Est. Time | Status |
|---|---|---|---|
| Run Naive Fine-Tuning baseline for BWT comparison | High | 2 hrs | Not started |
| Execute ablation study (V2 ready: `--force-validation`) | High | 2–3 hrs | Implemented ✓ |
| Multi-seed validation — 5 seeds, CIs, t-tests (V2 ready) | High | 2–3 hrs | Implemented ✓ |
| Hyperparameter sensitivity sweeps (V2 ready) | Medium | 1–2 hrs | Implemented ✓ |
| Cross-app generalisation — 246K holdout rows (V2 ready) | Medium | ~30 min | Implemented ✓ |
| Full Kubernetes simulation with Minikube + pod scaling | Medium | TBD | Not started |
| Dissertation write-up (Results, Discussion, Conclusion) | High | Ongoing | In progress |

---

## Slide 5 — Planned vs Actual Progress (1 min)

| Milestone | Planned (Gantt) | Actual | Status |
|---|---|---|---|
| Supervisor & title selection | Nov 2025 | Nov 2025 | On time |
| Proposal submission | Dec 2025 | Dec 2025 | On time |
| Literature review | Dec 2025 – Jan 2026 | Dec 2025 – Jan 2026 | On time |
| Resource gathering / dataset | Jan – Feb 2026 | Jan – Feb 2026 | On time |
| Tools & techniques selection | Feb – Mar 2026 | Feb – Mar 2026 | On time |
| Framework implementation (model + CL engine) | Mar – May 2026 | Mar – Apr 2026 | **Ahead** |
| Baseline comparison + core evaluation | Mar – May 2026 | Apr 2026 | **Ahead** |
| Bug fixes & V2 parameter corrections | Not explicitly planned | Apr 2026 | **Done** |
| Ablation + multi-seed + sensitivity analysis | Mar – May 2026 | Implemented in V2; run pending | **On track** |
| Kubernetes live cluster integration | Mar – May 2026 | **Not yet started** | Behind |
| Documentation writing | Apr – Jun 2026 | Starting May 2026 | On track |
| Publication / conference submission | May – Jun 2026 | Targeting Jun 2026 | On track |
| Presentation & manuscript submission | Jun 2026 | Jun 2026 | On track |

### Summary
The implementation phase is ahead of schedule — the full training pipeline, baselines, and initial evaluation are complete. The slight delay is in the validation depth (ablation, multi-seed, sensitivity) rather than the core implementation. These are being run now after the V2 parameter corrections. The overall trajectory remains on schedule for June 2026 submission.

---

## Anticipated Q&A — Prepared Answers

**Q: Why is your HPA baseline result showing near-zero MAE?**  
A: That was a bug in V1 — the HPA predictor used a 1-step lag (5 minutes) while neural models predict 30 minutes ahead, making HPA artificially appear best. This has been corrected in V2: HPA now uses the same 6-step forecast horizon. The corrected HPA MAE is 1.26 (still lower than neural models, as expected — HPA is a reactive system with full current-state observability, whereas our model predicts 30 minutes ahead without knowing current values).

**Q: What does BWT = -240 mean — is your model forgetting?**  
A: BWT measures performance change on earlier chunks after training on later ones. For MAE (where lower is better), a negative BWT means the model improved on old data — which is positive. In our V2 results, Chunk 1 MAE improved 53% after all chunks were trained (positive backward transfer). Chunks 2 and 3 degraded modestly (+8% and +15% respectively). So BWT = -240 is actually a good result: the model learned from later data in a way that helped it generalise to earlier workload patterns, with only minor degradation on intermediate chunks. Compared to V1 (where Chunk 3 degraded 39%), V2 shows substantially less forgetting thanks to the higher EWC lambda (500) and larger replay buffer (2000 samples).

**Q: How does your approach differ from simply retraining periodically?**  
A: Periodic retraining forgets all earlier patterns each cycle. Our EWC penalty mathematically protects important weights, and Experience Replay ensures the model revisits past samples. The goal is to achieve better accuracy on the *full history* of workloads, not just the most recent chunk. The dissertation will compare BWT across both methods to demonstrate this.

**Q: Why use the Alibaba dataset instead of real CI/CD pipeline data?**  
A: The Alibaba Cluster Traces are publicly available, large-scale production data covering diverse workload types including batch jobs, GPU workloads, and long-running services. Real CI/CD data is proprietary and difficult to obtain at scale. The dataset's heterogeneity makes it a reasonable proxy, and the framework is designed to generalise to CI/CD-specific patterns.

**Q: What is the practical significance of your SLA violation rate?**  
A: The proposed model achieves 0.30% SLA violations — essentially matching the reactive Kubernetes HPA baseline (0.25%), while predicting 30 minutes ahead rather than reacting after threshold breach. Periodic Retraining achieves 7.43% and Static LSTM 8.38%. Our model is 24× better than Periodic Retrain on this metric. In a production CI/CD pipeline, every SLA violation means a build slows down or fails due to CPU starvation. Reducing violations from 7–8% to 0.30% translates to roughly 96% fewer resource-starvation events.

---

*Presentation duration target: 5 minutes. Practice recommended at 1 min/slide.*
