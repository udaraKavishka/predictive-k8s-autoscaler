# GCP Kubernetes Research Guide
## Adaptive CPU Auto-Scaling — Kubernetes Simulation on Google Cloud

**Research:** Adaptive CPU Auto-Scaling Using Hybrid Neural Networks and Continual Learning in Kubernetes Orchestration
**Author:** Udara Kabishka Nalawansa
**Purpose:** Run the Kubernetes simulation phase on GCP for dissertation evidence

---

## Table of Contents

0. [Phase 0 — Teardown Existing Cluster (if restarting)](#0-phase-0--teardown-existing-cluster)
1. [Prerequisites](#1-prerequisites)
2. [GCP Free Tier Overview](#2-gcp-free-tier-overview)
3. [Phase 1 — GCP & GKE Setup](#3-phase-1--gcp--gke-setup)
4. [Phase 2 — Install Prometheus & Grafana](#4-phase-2--install-prometheus--grafana)
5. [Phase 3 — Build & Push Autoscaler Image](#5-phase-3--build--push-autoscaler-image)
6. [Phase 4 — Upload Model & Data](#6-phase-4--upload-model--data)
7. [Phase 5 — Deploy All Components](#7-phase-5--deploy-all-components)
8. [Phase 6 — Run the Simulation](#8-phase-6--run-the-simulation)
9. [Phase 7 — Collect Dissertation Evidence](#9-phase-7--collect-dissertation-evidence)
10. [Troubleshooting](#10-troubleshooting)
11. [Cleanup](#11-cleanup)

---

## 0. Phase 0 — Teardown Existing Cluster (if restarting)

If you already created a cluster and want to restart with a new configuration (e.g., changing from 2 nodes to 3 nodes), delete all resources first:

```bash
# Delete the GKE cluster (stops all VM billing immediately)
gcloud container clusters delete research-autoscaler \
  --zone us-central1-a \
  --quiet

# Delete Artifact Registry (frees up storage quota)
gcloud artifacts repositories delete research \
  --location=us-central1 \
  --quiet

# Wait for deletion to complete (~2 minutes)
# Verify:
gcloud container clusters list
gcloud artifacts repositories list --location=us-central1
```

> **Important:** Deleting the cluster stops billing for the compute nodes. Proceed to Phase 1 to recreate with the new configuration.

---

## 1. Prerequisites

Install these tools on your local machine before starting.

```bash
# Google Cloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud version   # should show 450+

# kubectl
gcloud components install kubectl

# Helm (for Prometheus)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
helm version

# Docker (for building the autoscaler image)
# Install Docker Desktop or docker-ce for your OS

# Python 3.10+ (for local testing)
python3 --version
```

---

## 2. GCP Free Tier Overview

| Resource | Free Tier Detail | Cost if Exceeded |
|----------|-----------------|-----------------|
| GKE cluster management | **1 zonal Standard cluster FREE** (waived $0.10/hr fee) | $0.10/hr |
| Compute nodes | NOT free | Pay per VM |
| e2-small (2 vCPU, 2 GB) | ~$14/month each | ~$0.02/hr each |
| 3× e2-small nodes | ~$42/month | ~$0.06/hr total |
| Artifact Registry | 0.5 GB free | $0.10/GB/month |
| Network egress | 1 GB free/month | $0.12/GB |
| **$300 free credit** | **New accounts, 90 days** | — |

**Bottom line:**
- With the $300 credit, 3× e2-small nodes cost ~$42/month → credit lasts ~7 months.
- A 2-hour research simulation costs under $0.12.
- **Log aggregation:** This guide uses **Loki + Promtail** for centralized logging (Phase 2.5), NOT GCP Cloud Logging. Loki is open-source and runs in-cluster at **$0 extra cost**, saving ~$0.50/GB that GCP Logging would charge. `--no-enable-cloud-logging` was set intentionally.
- **Do NOT use Autopilot** — it enforces minimum resource requests per pod (0.5 vCPU, 2 GB RAM) which rapidly inflates cost. Use **GKE Standard**.
- **Do NOT use AWS EKS** — charges $0.10/hr (~$72/month) just for the control plane with no free tier.

---

## 3. Phase 1 — GCP & GKE Setup

### 3.1 Create a GCP Project

```bash
# Log in
gcloud auth login

# Create project (or use an existing one)
gcloud projects create research-autoscaler-2026 --name="Research Autoscaler"
gcloud config set project research-autoscaler-2026

# Link billing account (required even for free tier / credit)
# Find your billing account ID:
gcloud billing accounts list
# Link it:
gcloud billing projects link research-autoscaler-2026 \
  --billing-account=XXXXXX-XXXXXX-XXXXXX
```

### 3.2 Enable Required APIs

```bash
gcloud services enable \
  container.googleapis.com \
  artifactregistry.googleapis.com \
  cloudresourcemanager.googleapis.com \
  monitoring.googleapis.com
```

### 3.3 Create the GKE Cluster

```bash
# Single-zone cluster (cheapest — no redundancy needed for research)
# 3 nodes for fault tolerance (supervisor requirement)
# us-central1-a qualifies for free e2-micro, but we use e2-small for performance
gcloud container clusters create research-autoscaler \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type e2-small \
  --disk-size 20 \
  --no-enable-autoscaling \
  --no-enable-autorepair \
  --release-channel None \
  --no-enable-cloud-logging \
  --no-enable-cloud-monitoring

# This creates 1 free zonal cluster (management fee waived)
# Estimated cost: ~$0.06/hr for the 3 e2-small nodes (~$42/month)
```

### Pod Capacity with 3 × e2-small

```bash
# Resource capacity verification:
#   Total CPU:     6 vCPU   → supports ~60 nginx pods at 100m CPU each
#   Total RAM:     6 GB     → well above the 10-pod MAX_REPLICAS ceiling
#   GKE pod limit: 110 pods per node → 330 pods maximum (never a constraint)
# 
# Conclusion: MAX_REPLICAS=10 is well within all resource and quota limits.
```

### 3.4 Configure kubectl

```bash
gcloud container clusters get-credentials research-autoscaler --zone us-central1-a
kubectl cluster-info
kubectl get nodes    # should show 3 nodes in Ready state
```

### 3.5 Pod Limits — FAQ

**Q: How many pods can GKE run on this cluster?**

GKE enforces **two independent ceilings**, and the smaller one wins:

| Limit | Value | Why |
|-------|-------|-----|
| GKE control plane limit | 110 pods/node | Default GKE quota |
| **Resource constraint** | ~20 pods/node | 2 vCPU per node ÷ 0.1 vCPU per pod |
| **Effective limit (3 nodes)** | **~60 pods total** | Resources are the bottleneck |
| Research `MAX_REPLICAS` | 10 pods | autoscaler config |

**Conclusion:** The autoscaler's `MAX_REPLICAS=10` is **well below the effective limit**. No scaling constraints.

---

## 4. Phase 2 — Install Prometheus & Grafana

```bash
# Add Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Create monitoring namespace
kubectl create namespace monitoring

# Install kube-prometheus-stack (Prometheus + Grafana + node-exporter + kube-state-metrics)
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  -n monitoring \
  -f k8s/prometheus-values.yaml

# Wait for all pods to be ready (takes ~2 minutes)
kubectl get pods -n monitoring -w
# Press Ctrl+C when all are Running

# Verify Prometheus is up
kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090 &
# Open: http://localhost:9090
# Run query: up  — should show several targets
```

### Access Grafana

```bash
kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80 &
# Open: http://localhost:3000
# Login: admin / research2026
```

---

## 4.5 Phase 2.5 — Install Loki + Promtail (Log Aggregation)

Centralized log collection for dissertation evidence (scaling decisions, nginx access logs, load generator activity).
**Cost:** $0 extra (runs in cluster on existing nodes). Replaces GCP Cloud Logging which is disabled for cost savings.

> **Note:** `grafana/loki-stack` chart is deprecated. We use the new `grafana/loki` (single-binary) + `grafana/promtail` (separate chart).

```bash
# Add Grafana Helm repo
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

cd /home/udara/Documents/Research/k8s_autoscaler

# ONLY if you previously installed loki-stack (deprecated), uninstall it first:
helm uninstall loki -n monitoring   # skip if this is your first install

# Step 1: Install Loki (single-binary mode)
helm install loki grafana/loki \
  -n monitoring \
  -f k8s/loki-values.yaml

# Step 2: Install Promtail (log collector DaemonSet)
helm install promtail grafana/promtail \
  -n monitoring \
  -f k8s/promtail-values.yaml

# Wait for all pods (takes ~2 minutes)
kubectl get pods -n monitoring -w
# Press Ctrl+C when you see:
#   loki-0                          1/1   Running
#   promtail-xxxxx (×3)             1/1   Running   ← one per node
```

### Add Loki as a Grafana Data Source

```bash
# Keep Grafana port-forward running (from Phase 2)
# Open: http://localhost:3000 → Configuration → Data Sources → Add data source

# Or use the Grafana CLI:
kubectl exec -n monitoring deployment/kube-prometheus-stack-grafana -- \
  grafana-cli admin provisioning datasources
```

**In Grafana UI:**
1. Click **Configuration** (gear icon)
2. Select **Data Sources**
3. Click **Add data source**
4. Type: **Loki**
5. URL: `http://loki.monitoring.svc.cluster.local:3100`
6. Click **Save & Test** → should show "Data source connected"

### Verify Promtail is Collecting Logs

```bash
# Check Promtail pods (one per node)
kubectl get pods -n monitoring -l app.kubernetes.io/name=promtail -o wide
# Expected output: 3 pods, one on each node

# Check Loki is ready
kubectl port-forward -n monitoring svc/loki 3100:3100 &
curl http://localhost:3100/ready
# Expected output: "ready"

# Verify Promtail is sending logs to Loki
kubectl logs -n monitoring -l app.kubernetes.io/name=promtail --tail=5
# Should show messages like "Grafana Loki client" or successful push events
```

---

## 5. Phase 3 — Build & Push Autoscaler Image

**Note on TensorFlow installation:**
The Dockerfile installs TensorFlow (2.17.0) and other heavy ML dependencies directly in a separate RUN layer with an extended pip timeout (600 seconds). This prevents socket timeouts when downloading large wheels (~475 MB for TensorFlow).
`requirements.txt` contains only the lightweight runtime dependencies (pandas, kubernetes, requests, prometheus-api-client).

```bash
# Set your project ID
export PROJECT_ID=research-autoscaler-2026
export REGION=us-central1

# Create Artifact Registry repository
gcloud artifacts repositories create research \
  --repository-format=docker \
  --location=$REGION

# Configure Docker to authenticate with GCP
gcloud auth configure-docker $REGION-docker.pkg.dev

# Build the image (from k8s_autoscaler directory)
cd /home/udara/Documents/Research/k8s_autoscaler
docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/research/autoscaler:v1 .

# Push to Artifact Registry
docker push $REGION-docker.pkg.dev/$PROJECT_ID/research/autoscaler:v1

# Verify push
gcloud artifacts docker images list $REGION-docker.pkg.dev/$PROJECT_ID/research
```

### Update Image Reference in Manifests

Replace the placeholder image in the YAML files:

```bash
sed -i "s|REGION-docker.pkg.dev/PROJECT_ID|$REGION-docker.pkg.dev/$PROJECT_ID|g" \
  k8s/autoscaler-cronjob.yaml k8s/load-generator-job.yaml
```

---

## 6. Phase 4 — Upload Model & Data

### 6.1 Upload Trained Model + Scalers

Your research code (`research_imp_V2.py`) already produced these files in the output
directory after training. You need to upload all four:

| File | What it is |
|------|-----------|
| `hybrid_model_ewc_er.keras` | The trained LSTM+MLP model (after all 4 continual learning chunks) |
| `temporal_scaler.pkl` | StandardScaler fitted on temporal features (must be applied before inference) |
| `static_scaler.pkl` | StandardScaler fitted on static features |
| `target_scaler.pkl` | StandardScaler fitted on cpu_demand (used to inverse-transform predictions) |

```bash
# Apply namespaces and PVCs first
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/autoscaler-cronjob.yaml  # creates model-pvc

# Wait for PVC to be bound
kubectl get pvc -n research-autoscaler

# If model-pvc already exists with the old access mode, delete and re-create it first:
# kubectl delete pvc model-pvc -n research-autoscaler
# kubectl apply -f k8s/autoscaler-cronjob.yaml

# Create a temporary pod to upload files
kubectl run model-uploader \
  --image=busybox \
  --restart=Never \
  -n research-autoscaler \
  --overrides='{"spec":{"volumes":[{"name":"model","persistentVolumeClaim":{"claimName":"model-pvc"}}],"containers":[{"name":"busybox","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"model","mountPath":"/model"}]}]}}'

kubectl wait pod/model-uploader -n research-autoscaler --for=condition=Ready --timeout=60s

# Set this to the directory where research_imp_V2.py saved its output
set RESEARCH_OUTPUT_DIR "/home/udara/Documents/Research/"

# Copy all four required files
kubectl cp $RESEARCH_OUTPUT_DIR/hybrid_model_ewc_er.keras \
  research-autoscaler/model-uploader:/model/hybrid_model_ewc_er.keras
kubectl cp $RESEARCH_OUTPUT_DIR/temporal_scaler.pkl \
  research-autoscaler/model-uploader:/model/temporal_scaler.pkl
kubectl cp $RESEARCH_OUTPUT_DIR/static_scaler.pkl \
  research-autoscaler/model-uploader:/model/static_scaler.pkl
kubectl cp $RESEARCH_OUTPUT_DIR/target_scaler.pkl \
  research-autoscaler/model-uploader:/model/target_scaler.pkl

# Verify all 4 files are present
kubectl exec -n research-autoscaler model-uploader -- ls -lh /model/
# Expected output:
#   hybrid_model_ewc_er.keras   ~50 MB
#   temporal_scaler.pkl         ~few KB
#   static_scaler.pkl           ~few KB
#   target_scaler.pkl           ~few KB

kubectl delete pod model-uploader -n research-autoscaler
```

### 6.2 Upload Alibaba Trace Data

```bash
# Apply data PVC
kubectl apply -f k8s/load-generator-job.yaml  # creates data-pvc

# Create temp pod for upload
kubectl run data-uploader \
  --image=busybox \
  --restart=Never \
  -n research-workload \
  --overrides='{"spec":{"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"data-pvc"}}],"containers":[{"name":"busybox","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"data","mountPath":"/data"}]}]}}'

kubectl wait pod/data-uploader -n research-workload --for=condition=Ready --timeout=60s

# Copy the Alibaba time-series CSV
kubectl cp /home/udara/Documents/Research/alibaba_timeseries_full.csv \
  research-workload/data-uploader:/data/alibaba_timeseries_full.csv

kubectl exec -n research-workload data-uploader -- ls -lh /data/
kubectl delete pod data-uploader -n research-workload
```

---

## 7. Phase 5 — Deploy All Components

Apply manifests in this exact order:

```bash
cd /home/udara/Documents/Research/k8s_autoscaler

# 1. Namespaces (already applied above)
kubectl apply -f k8s/namespace.yaml

# 2. RBAC
kubectl apply -f k8s/rbac.yaml

# 3. ConfigMap
kubectl apply -f k8s/configmap.yaml

# 4. Sample workload (nginx + stress-ng)
kubectl apply -f k8s/sample-workload.yaml

# 5. Autoscaler CronJob (already applied above for PVC — re-apply is safe)
kubectl apply -f k8s/autoscaler-cronjob.yaml

# Verify everything is running
kubectl get all -n research-workload
kubectl get all -n research-autoscaler
kubectl get cronjobs -n research-autoscaler
```

Expected output:
```
NAME                          READY   UP-TO-DATE   AVAILABLE
deployment/sample-workload    1/1     1            1
deployment/load-generator     1/1     1            1

NAME                         SCHEDULE      SUSPEND   ACTIVE
cronjob/predictive-autoscaler */5 * * * *  False     0
```

---

## 8. Phase 6 — Run the Simulation

### 8.1 Trigger the Autoscaler Manually (First Test)

```bash
# Manually trigger one CronJob run to verify it works
kubectl create job --from=cronjob/predictive-autoscaler manual-test-1 \
  -n research-autoscaler

# Watch job status
kubectl get jobs -n research-autoscaler -w

# View logs
kubectl logs -n research-autoscaler job/manual-test-1 -f
```

Expected log output (v2, with the sim-to-real domain adapter):
```
[INFO] Loading model from /model/hybrid_model_ewc_er.keras
[INFO] Model weights loaded successfully
[INFO] Model and all three scalers loaded
[INFO] Domain adapter: scale=9058.3 (train mean 2717.5 / live ref 0.3)
[INFO] Kubernetes: using in-cluster config
[INFO] Predicted t+30min CPU: 0.1710 vCPU (+15% -> 0.1967) -> 2 replicas
[INFO] DECISION: {"timestamp":"...","current_cpu":0.1268,"predicted_cpu":0.1710,"old_replicas":2,"new_replicas":2,"method":"predictive"}
```

> **What "predictive" means here:** the model was trained on aggregated Alibaba `cpu_demand`
> (centered ~2717), while the live workload runs at ~0.1–1 vCPU. The **domain adapter** maps the
> live signal onto the training scale before inference and back afterwards (`LIVE_REF_CPU=0.3` →
> `DOMAIN_SCALE≈9058`). Without it, every prediction lands in the hundreds and the run logs
> `reactive-fallback-sanity` instead of `predictive`. `LIVE_REF_CPU` is set in `k8s/configmap.yaml`.

### 8.2 Drive the Load with `ramp_demo.sh` (the real evidence run)

`ramp_demo.sh` is the actual driver of the simulation. It steps the `load-generator`
stress-ng `--cpu-load` through phases so the predictive autoscaler reads a rising/falling
CPU signal from Prometheus, predicts 30 min ahead, and patches the **`load-generator`**
deployment — producing real replica transitions. (The autoscaler target is `load-generator`,
set in `k8s/configmap.yaml`; `sample-workload` is a static nginx that stays at 1 replica.)

```bash
cd /home/udara/Documents/Research/k8s_autoscaler

# Full run: 2-hr history flush → ramp up 1→2→3 → ramp down 3→2→1, then auto-collect
bash ramp_demo.sh

# Useful flags:
bash ramp_demo.sh --no-flush   # skip the 2-hr flush (cluster already idle/low)
bash ramp_demo.sh --up-only    # ramp up only (1→2→3), no scale-down phase
bash ramp_demo.sh --quick      # 10-min phases — tests script logic, NOT real evidence
```

CPU-load → expected replica mapping (`CPU_PER_REPLICA=0.1`, `SLA_TOLERANCE=0.15`):

| `--cpu-load` | avg CPU/pod | desired replicas |
|--------------|-------------|------------------|
| 25% | ~0.050 vCPU | 1 |
| 50% | ~0.100 vCPU | 2 |
| 80% | ~0.160 vCPU | 3 |

Each phase holds for 25 min = **5 autoscaler cycles**, so every replica level is *sustained*
across several decisions rather than a single blip. With the raised CronJob history limit
(`successfulJobsHistoryLimit: 60` in `k8s/autoscaler-cronjob.yaml`), the **entire** ramp
survives in `decisions.csv` — the earlier 12-run limit truncated the scale-up.

`ramp_demo.sh` calls `collect_evidence.py` automatically at the end, so a single full run
produces the predictive-pass evidence in `evidence/`.

### 8.3 Watch it Live (optional, while the ramp runs)

All dissertation figures are generated by the Phase 7 collector from Prometheus data —
**not** from Grafana screenshots. Grafana is only useful here for an at-a-glance live view.

```bash
# Terminal 1: watch the load-generator replica count change
kubectl get pods -n research-workload -w

# Terminal 2: watch autoscaler jobs fire (every 5 min)
kubectl get jobs -n research-autoscaler -w

# Terminal 3: tail the decisions as they are written
tail -f evidence/decisions.jsonl
```

> **Note:** Grafana's `Compute Resources / Workload` dashboard has **no replica-count panel** —
> the replica-count evidence comes from `prometheus_replicas.json` + `decisions.csv` (Phase 7),
> rendered as `fig2_replica_count.png`. Do not look for a "Current Replicas" panel in Grafana.

### 8.4 HPA Reactive Baseline (same trace, for comparison)

To close the predictive-vs-reactive gap, run the **same ramp** under the stock Kubernetes
Horizontal Pod Autoscaler instead of the predictive CronJob, then collect a second labelled
evidence set. Phase 7 (`step7_compare`) turns the two passes into a side-by-side result.

```bash
# 1. Suspend the predictive autoscaler so the two controllers don't fight
kubectl patch cronjob predictive-autoscaler -n research-autoscaler \
  -p '{"spec":{"suspend":true}}'

# 2. Create a reactive HPA on the SAME deployment (CPU target ~80%)
kubectl autoscale deployment load-generator -n research-workload \
  --cpu-percent=80 --min=1 --max=10

# 3. Run the identical ramp (the HPA now does the scaling)
bash ramp_demo.sh --no-flush

# 4. Collect this pass into evidence/hpa/  (see §9)
EVIDENCE_LABEL=hpa python3 collect_evidence.py

# 5. Tear down the HPA and re-enable the predictive autoscaler
kubectl delete hpa load-generator -n research-workload
kubectl patch cronjob predictive-autoscaler -n research-autoscaler \
  -p '{"spec":{"suspend":false}}'
```

> Run the **predictive** pass the same way but with `EVIDENCE_LABEL=predictive` (§9), so the
> two labelled folders line up for the comparison.

---

## 9. Phase 7 — Collect Dissertation Evidence

### 9.1 Run the Evidence Collector

The **core evidence is produced entirely by two Python scripts** — Grafana screenshots
(§9.4) are an optional visual supplement, not the source of any data:

| Script | Role | Produces |
|--------|------|----------|
| `ramp_demo.sh` | Drives the load through the ramp (calls the collector at the end) | the predictive-pass run |
| `collect_evidence.py` | Pulls autoscaler logs, decisions, Prometheus time-series, cluster state; writes the summary; renders the figures | everything in `evidence/` (see §9.2) |

Keep a Prometheus port-forward running first, then run the collector:

```bash
kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090 &

cd /home/udara/Documents/Research/k8s_autoscaler

# Predictive pass → evidence/predictive/
EVIDENCE_LABEL=predictive python3 collect_evidence.py

# HPA pass (after running §8.4) → evidence/hpa/
EVIDENCE_LABEL=hpa python3 collect_evidence.py
```

For a single quick run (no comparison) just `python3 collect_evidence.py` → writes to
`evidence/`. Note `ramp_demo.sh` already calls the collector at the end of a predictive run.

> **Collect before jobs are garbage-collected.** The CronJob keeps
> `successfulJobsHistoryLimit: 60` runs (~5 h). Run the collector during or right after each
> ramp — if more than ~5 h of runs accumulate, the oldest decisions drop out of
> `decisions.csv` (this is what truncated the earlier 12-run evidence).

### 9.2 Evidence Artifacts — What Each File Shows

Everything below lands in `evidence/` (or `evidence/<label>/` for a labelled pass):

| Artifact | What it shows | Paper mapping |
|----------|---------------|---------------|
| `decisions.csv` / `.jsonl` | Every autoscaler decision: `current_cpu`, `predicted_cpu`, `old/new_replicas`, `method` | §6.7 decision table; predictive vs reactive count |
| `summary_report.txt` | Totals: # decisions, % predictive vs reactive, max/min replicas, scale-up/down counts | §6.7 narrative |
| `prometheus_load_generator_cpu.json` | avg load-generator CPU time-series (the signal driving decisions) | source for `fig1` |
| `prometheus_replicas.json` | replica count time-series per deployment (the scaling response) | source for `fig2` |
| `prometheus_cpu.json` | per-pod CPU time-series | supporting detail |
| `scaling_events.txt` | `kubectl` scaling events in `research-workload` | §6.7 evidence of real K8s actions |
| `pod_status.txt`, `job_history.txt`, `deployment_status.txt` | cluster-state snapshots (pods, autoscaler jobs, deployment replicas) | methodology / appendix |
| `fig1_cpu_demand_curve.png` | load-generator CPU vs time with replica thresholds | §6.7 Figure — CPU demand |
| `fig2_replica_count.png` | replica step plot (1→2→3→2→1) | §6.7 Figure — scaling response |
| `fig3_autoscaler_decisions.png` | current vs predicted CPU + replicas (predictive pass only) | §6.7 Figure — prediction quality |
| `comparison.csv`, `fig4_predictive_vs_hpa.png` | predictive vs HPA SLA-violation % and over-provision % (two-pass only) | §6.7 live baseline (§9.5) |

> **The replica-count evidence is `prometheus_replicas.json` + `fig2`** — *not* a Grafana
> dashboard panel. No prebuilt Grafana dashboard plots replica count (see §9.4 for why, and
> how to build a custom panel if you want one).

### 9.3 Prometheus Queries (manual spot-checks)

The collector runs these for you; use them only to sanity-check live in the Prometheus UI
(`http://localhost:9090`):

```promql
# avg load-generator CPU (the signal the autoscaler reads)
avg(rate(container_cpu_usage_seconds_total{namespace="research-workload",pod=~"load-generator-.*",container!="POD",container!=""}[300s]))

# Replica count over time — note the target is load-generator, not sample-workload
kube_deployment_spec_replicas{namespace="research-workload",deployment="load-generator"}
```

### 9.4 Grafana Screenshots (optional supplementary visuals)

These are **not** a data source — every number and figure already comes from §9.1. Take
these only if you want polished dashboard visuals to sit alongside the script figures.

```bash
kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80 &
# Open http://localhost:3000  (admin / research2026)
```

Set the time range (top-right clock icon) to your simulation window (e.g. `Last 3 hours`)
before every screenshot.

**Screenshots that genuinely exist as prebuilt panels:**

| # | Dashboard → panel | Filter | What it shows | Save as |
|---|-------------------|--------|---------------|---------|
| 1 | `Kubernetes / Compute Resources / Namespace (Workloads)` → **CPU Usage** | namespace = `research-workload` | load-generator CPU curve following the ramp | `grafana_cpu_namespace.png` |
| 2 | `Kubernetes / Compute Resources / Namespace (Pods)` → **CPU Usage** | namespace = `research-workload` | per-pod CPU; load-generator pods appear/disappear as it scales | `grafana_cpu_pods.png` |
| 3 | `Kubernetes / Compute Resources / Cluster` → **CPU Utilisation** + **CPU Requests Commitment** | *(none)* | cluster ran under real pressure | `grafana_cluster_util.png` |
| 4 | **Explore** → Loki data source → `{namespace="research-autoscaler"} \|= "DECISION"` | `Last 3h` | the DECISION JSON log lines (method, predicted_cpu, replicas) | `grafana_decision_logs.png` |

**⚠ Replica count is NOT a prebuilt Grafana panel.** None of the Compute Resources dashboards
— *including `Compute Resources / Workload`* — plot replica count. You have two options:

- **Recommended:** just use the script figure **`fig2_replica_count.png`** (already generated
  from `prometheus_replicas.json`).
- **Or build a custom panel** if you want the replica curve inside Grafana: *New panel* →
  paste PromQL
  `kube_deployment_spec_replicas{namespace="research-workload",deployment="load-generator"}`
  → visualisation **Time series** (or **State timeline**) → screenshot as
  `grafana_replicas_custom.png`. This is the only way to get a replica chart from Grafana.

**Steps to capture any panel:**
1. Open the dashboard; set the namespace filter and time range.
2. Hover the panel → click the `⋮` menu in its top-right corner.
3. **Share → Direct link rendered image** → if a *Render image* button appears, click it to
   download the PNG directly.
4. If the image-renderer plugin isn't installed, instead press `v` (or panel title → **View**)
   to expand the panel full-screen, then take an OS screenshot
   (Linux: `gnome-screenshot -a` for an area grab).
5. Save it under the name from the table above.

### 9.5 Predictive vs HPA Comparison

After both labelled passes exist (`evidence/predictive/` and `evidence/hpa/` — see §8.4 and
§9.1), the collector's comparison step emits the live baseline result automatically:

- **`evidence/comparison.csv`** — one row per method with `sla_violation_pct`,
  `over_provision_pct`, `mean_cpu`, `mean_replicas`, `n_points`.
- **`evidence/fig4_predictive_vs_hpa.png`** — grouped bar chart of SLA-violation % and
  over-provision % for predictive vs HPA.

These are computed from each pass's `prometheus_replicas.json` + load-generator CPU, using
provisioned CPU = `replicas × CPU_PER_REPLICA` (0.1): an **SLA violation** is a sample where
actual CPU exceeds provisioned, and **over-provision %** is the mean headroom
`(provisioned − actual) / actual`. This is the live, on-cluster predictive-vs-HPA comparison
that fills the `Final_Research.md §6.7` baseline gap.

Before citing an `evidence/hpa/` folder as Kubernetes HPA evidence, verify that the predictive
CronJob was disabled during that pass and save the proof next to the run:

- `kubectl get hpa -n research-workload -o yaml > evidence/hpa/hpa.yaml`
- `kubectl get hpa -n research-workload > evidence/hpa/hpa_status.txt`
- `kubectl get cronjob -n research-autoscaler > evidence/hpa/autoscaler_cronjob_status.txt`

If `evidence/hpa/decisions.csv` contains `method=predictive`, do not use those decision rows as
HPA decisions. In that case, only the Prometheus replica and CPU time series can be cited, and the
run should be described as an over-provisioned or reactive baseline unless the saved HPA status
proves that Kubernetes HPA controlled the deployment.

> If `comparison.csv` is missing, one of the two labelled folders (or its
> `prometheus_replicas.json`) doesn't exist — re-run the missing pass (§8.4).

### 9.6 Manual Decision-Log Export (fallback only)

`collect_evidence.py` already does this. Use it only if the script can't run (e.g. no
Python on the box) — it pulls logs from every autoscaler job and rebuilds `decisions.csv`:

```bash
# Concatenate logs from all completed autoscaler jobs
# (fish shell: keep the bash -c wrapper so the loop runs under bash)
bash -c 'for job in $(kubectl get jobs -n research-autoscaler -o name); do kubectl logs -n research-autoscaler $job >> evidence/autoscaler_decisions.log 2>/dev/null; done'

# Extract the DECISION JSON lines
grep "DECISION:" evidence/autoscaler_decisions.log | sed 's/.*DECISION: //' > evidence/decisions.jsonl
```

Then feed `evidence/decisions.jsonl` into the same CSV conversion the script uses.

---

## 10. Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| CronJob never fires | Cluster clock drift | `kubectl get nodes -o wide` — check time sync |
| "No model found" in logs | Model not mounted correctly | `kubectl exec -n research-autoscaler` into a pod, check `/model/` |
| "Prometheus query failed" | Wrong Prometheus URL in ConfigMap | `kubectl get svc -n monitoring` — find correct service name |
| Pods stuck in `Pending` | Insufficient node resources | `kubectl describe pod <name>` — check events; reduce resource requests |
| "cannot patch deployments" | RBAC not applied | `kubectl apply -f k8s/rbac.yaml` |
| PVC stuck in `Pending` | Storage class not available | `kubectl get storageclass` — check available classes |
| `ImagePullBackOff` | Wrong image path or auth | Check image URL in manifest matches Artifact Registry |
| `INSTALLATION FAILED: cannot re-use a name` | Helm release already exists from previous attempt | `helm upgrade <name> <chart> -n <namespace> -f <values>` to update, or `helm uninstall` then reinstall |
| Autoscaler predicts 0 replicas | Near-zero CPU in history | Ensure `MIN_REPLICAS=1` in ConfigMap; start load generator first |
| Every run logs `reactive-fallback-sanity` / predicted ~hundreds of vCPU | Train/live CPU scale mismatch (model trained on aggregated Alibaba demand ~2717, live workload ~0.1 vCPU) | Use the v2 image (domain adapter on). Tune `LIVE_REF_CPU` in `k8s/configmap.yaml` to the workload's typical vCPU so predictions land in range. NOT a sklearn version problem |

### Helm Release Already Exists

If you encounter `Error: INSTALLATION FAILED: cannot re-use a name that is still in use` when installing Prometheus or Loki:

```bash
# For kube-prometheus-stack (if already installed):
helm upgrade kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  -n monitoring -f k8s/prometheus-values.yaml

# For loki (if already installed):
helm upgrade loki grafana/loki-stack \
  -n monitoring -f k8s/loki-values.yaml

# Alternative: uninstall and reinstall (clean slate)
helm uninstall kube-prometheus-stack -n monitoring
# wait 30 seconds, then reinstall
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  -n monitoring -f k8s/prometheus-values.yaml
```

> Use `helm upgrade` for faster deployment. Only uninstall+reinstall if pods are in a broken state.

---

## 11. Cleanup

**Delete all research resources (stops all billing for nodes):**

```bash
# Delete the GKE cluster (this stops VM billing immediately)
gcloud container clusters delete research-autoscaler \
  --zone us-central1-a \
  --quiet

# Delete Artifact Registry (optional — only 0.5 GB free)
gcloud artifacts repositories delete research \
  --location=us-central1 \
  --quiet

# (Optional) Delete the project entirely
gcloud projects delete research-autoscaler-2026
```

> **Important:** Deleting the cluster stops all node billing. The project itself has no recurring cost once the cluster is deleted.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    GKE Cluster (3× e2-small)                │
│                                                             │
│  ┌─────────────────┐      ┌─────────────────────────────┐  │
│  │  research-       │      │  research-autoscaler         │  │
│  │  workload ns     │      │  namespace                   │  │
│  │                 │      │                              │  │
│  │  ┌───────────┐  │      │  ┌────────────────────────┐ │  │
│  │  │  sample-  │◄─┼──────┼──│  predictive-autoscaler  │ │  │
│  │  │  workload │  │patch │  │  CronJob (*/5 min)      │ │  │
│  │  │ (nginx)   │  │scale │  │                        │ │  │
│  │  │ 1-10 pods │  │      │  │  1. Query Prometheus    │ │  │
│  │  └───────────┘  │      │  │  2. Load LSTM+MLP model │ │  │
│  │                 │      │  │  3. Predict 30 min ahead│ │  │
│  │  ┌───────────┐  │      │  │  4. Patch Deployment    │ │  │
│  │  │  load-    │  │      │  └────────────────────────┘ │  │
│  │  │ generator │  │      │          │                   │  │
│  │  │(stress-ng)│  │      │     /model/saved_model.h5   │  │
│  │  └───────────┘  │      │     (PersistentVolumeClaim)  │  │
│  └─────────────────┘      └─────────────────────────────┘  │
│            │                                                │
│            │ metrics                                        │
│            ▼                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  monitoring namespace                                │   │
│  │  Prometheus ◄── node-exporter, kube-state-metrics   │   │
│  │  Grafana (live view only — figures come from the     │   │
│  │  collect_evidence.py script, not screenshots)        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

Local machine:
  ramp_demo.sh ───────► drives load-generator stress-ng CPU
                        through phases (1→2→3→2→1 replicas)
  collect_evidence.py ► pulls logs + Prometheus + renders figures
```

---

## Next Steps After Simulation

1. **Collect evidence** → `EVIDENCE_LABEL=predictive python3 collect_evidence.py` (decisions.csv + fig1–fig3)
2. **Run comparison** → HPA-Reactive baseline on the same ramp (§8.4) → `comparison.csv` + fig4
3. **Write dissertation section 6.7** — Kubernetes Simulation Results (fill the §6.7 baseline gap)
4. **Delete cluster** to stop billing
5. Remaining validation (ablation, multi-seed) runs on local Python, not GKE
