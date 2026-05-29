#!/usr/bin/env bash
# restart.sh — Recreate the research GKE cluster through Phase 7
# Usage: bash restart.sh
# Idempotent: safe to re-run if a previous attempt was interrupted.
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID=research-autoscaler-2026
REGION=us-central1
ZONE=us-central1-a
CLUSTER=research-autoscaler
IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/research/autoscaler:v1
RESEARCH_DIR=/home/udara/Documents/Research
K8S_DIR=$RESEARCH_DIR/k8s_autoscaler

echo "=========================================================="
echo " Research Autoscaler — GKE Restart Script"
echo " Project : $PROJECT_ID"
echo " Cluster : $CLUSTER ($ZONE)"
echo "=========================================================="
echo ""

# ── Phase 1 — GCP & GKE Setup ─────────────────────────────────────────────────
echo ">>> Phase 1: GCP & GKE setup"
gcloud config set project "$PROJECT_ID"

gcloud services enable \
  container.googleapis.com \
  artifactregistry.googleapis.com \
  cloudresourcemanager.googleapis.com \
  monitoring.googleapis.com

if gcloud container clusters describe "$CLUSTER" --zone "$ZONE" &>/dev/null; then
  echo "    Cluster already exists — skipping create"
else
  gcloud container clusters create "$CLUSTER" \
    --zone "$ZONE" \
    --num-nodes 3 \
    --machine-type e2-small \
    --disk-size 20 \
    --no-enable-autoscaling \
    --no-enable-autorepair \
    --release-channel None \
    --no-enable-cloud-logging \
    --no-enable-cloud-monitoring
fi

gcloud container clusters get-credentials "$CLUSTER" --zone "$ZONE"
kubectl cluster-info
kubectl get nodes

# ── Phase 2 — Prometheus + Grafana ───────────────────────────────────────────
echo ""
echo ">>> Phase 2: Prometheus + Grafana"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo update

kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

helm upgrade --install kube-prometheus-stack \
  prometheus-community/kube-prometheus-stack \
  -n monitoring \
  -f "$K8S_DIR/k8s/prometheus-values.yaml"

kubectl rollout status deployment/kube-prometheus-stack-grafana \
  -n monitoring --timeout=180s

# ── Phase 2.5 — Loki + Promtail ──────────────────────────────────────────────
echo ""
echo ">>> Phase 2.5: Loki + Promtail"
helm repo add grafana https://grafana.github.io/helm-charts 2>/dev/null || true
helm repo update

helm upgrade --install loki grafana/loki \
  -n monitoring \
  -f "$K8S_DIR/k8s/loki-values.yaml"

helm upgrade --install promtail grafana/promtail \
  -n monitoring \
  -f "$K8S_DIR/k8s/promtail-values.yaml"

kubectl rollout status daemonset/promtail -n monitoring --timeout=180s

# ── Phase 3 — Build & Push Autoscaler Image ───────────────────────────────────
# Always rebuilds — ensures any changes to autoscaler.py (sanity check, padding fix,
# etc.) are picked up. Image is ~847 MB; push takes ~5 min on a fast connection.
echo ""
echo ">>> Phase 3: Build & push autoscaler image (always rebuilds to pick up code fixes)"
if gcloud artifacts repositories describe research --location="$REGION" &>/dev/null; then
  echo "    Artifact Registry repo already exists — skipping create"
else
  gcloud artifacts repositories create research \
    --repository-format=docker \
    --location="$REGION"
fi

gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

cd "$K8S_DIR"
docker build -t "$IMAGE" .
docker push "$IMAGE"

# ── Phase 4 — Upload Model & Data ────────────────────────────────────────────
echo ""
echo ">>> Phase 4: Upload model & data"

# Apply namespaces + manifests to create PVCs
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/autoscaler-cronjob.yaml   # creates model-pvc
kubectl apply -f k8s/load-generator-job.yaml   # creates data-pvc

echo "    Waiting for PVCs to bind..."
kubectl wait --for=jsonpath='{.status.phase}'=Bound \
  pvc/model-pvc -n research-autoscaler --timeout=90s
kubectl wait --for=jsonpath='{.status.phase}'=Bound \
  pvc/data-pvc -n research-workload --timeout=90s

# Upload model artifacts
echo "    Uploading model files..."
kubectl run model-uploader \
  --image=busybox \
  --restart=Never \
  -n research-autoscaler \
  --overrides='{
    "spec": {
      "volumes": [{"name":"model","persistentVolumeClaim":{"claimName":"model-pvc"}}],
      "containers": [{
        "name": "busybox",
        "image": "busybox",
        "command": ["sleep","3600"],
        "volumeMounts": [{"name":"model","mountPath":"/model"}]
      }]
    }
  }' 2>/dev/null || echo "    model-uploader pod already exists"

kubectl wait pod/model-uploader -n research-autoscaler --for=condition=Ready --timeout=60s

kubectl cp "$RESEARCH_DIR/hybrid_model_ewc_er.keras" \
  research-autoscaler/model-uploader:/model/hybrid_model_ewc_er.keras
kubectl cp "$RESEARCH_DIR/temporal_scaler.pkl" \
  research-autoscaler/model-uploader:/model/temporal_scaler.pkl
kubectl cp "$RESEARCH_DIR/static_scaler.pkl" \
  research-autoscaler/model-uploader:/model/static_scaler.pkl
kubectl cp "$RESEARCH_DIR/target_scaler.pkl" \
  research-autoscaler/model-uploader:/model/target_scaler.pkl

echo "    Model files uploaded:"
kubectl exec -n research-autoscaler model-uploader -- ls -lh /model/
kubectl delete pod model-uploader -n research-autoscaler

# Upload trace data
echo "    Uploading Alibaba trace CSV..."
kubectl run data-uploader \
  --image=busybox \
  --restart=Never \
  -n research-workload \
  --overrides='{
    "spec": {
      "volumes": [{"name":"data","persistentVolumeClaim":{"claimName":"data-pvc"}}],
      "containers": [{
        "name": "busybox",
        "image": "busybox",
        "command": ["sleep","3600"],
        "volumeMounts": [{"name":"data","mountPath":"/data"}]
      }]
    }
  }' 2>/dev/null || echo "    data-uploader pod already exists"

kubectl wait pod/data-uploader -n research-workload --for=condition=Ready --timeout=60s

kubectl cp "$RESEARCH_DIR/alibaba_timeseries_full.csv" \
  research-workload/data-uploader:/data/alibaba_timeseries_full.csv

echo "    Data files uploaded:"
kubectl exec -n research-workload data-uploader -- ls -lh /data/
kubectl delete pod data-uploader -n research-workload

# ── Phase 5 — Deploy All Components ──────────────────────────────────────────
echo ""
echo ">>> Phase 5: Deploy all components"
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/rbac.yaml           # ServiceAccount + ClusterRoleBinding for both namespaces
kubectl apply -f k8s/configmap.yaml      # ConfigMap in both namespaces
kubectl apply -f k8s/sample-workload.yaml
kubectl apply -f k8s/autoscaler-cronjob.yaml

kubectl rollout status deployment/sample-workload -n research-workload --timeout=120s
kubectl rollout status deployment/load-generator  -n research-workload --timeout=120s

echo ""
kubectl get all -n research-workload
echo ""
kubectl get all -n research-autoscaler
echo ""
kubectl get cronjobs -n research-autoscaler

# ── Phase 6 — Manual verification test ───────────────────────────────────────
echo ""
echo ">>> Phase 6: Manual verification test"
kubectl delete job manual-test-1 -n research-autoscaler 2>/dev/null || true
sleep 3
kubectl create job --from=cronjob/predictive-autoscaler manual-test-1 \
  -n research-autoscaler

echo "    Waiting 60s for manual-test-1 to complete..."
sleep 60
echo "    --- manual-test-1 logs (last 20 lines) ---"
kubectl logs -n research-autoscaler job/manual-test-1 2>/dev/null | tail -20 || \
  echo "    (pod not yet ready — check manually with: kubectl logs -n research-autoscaler job/manual-test-1)"

# ── Phase 6 — Start trace replay (48 steps ≈ 24 min real time) ───────────────
echo ""
echo ">>> Phase 6: Start trace replayer (48 steps = 4 simulated hours at 10× speed)"
kubectl delete job trace-replayer -n research-workload 2>/dev/null || true
sleep 5
kubectl apply -f k8s/load-generator-job.yaml

echo ""
echo "    Trace replayer started."
echo "    The autoscaler monitors 'load-generator' CPU — it needs 24 × 5-min samples"
echo "    (~2 hours) before predictive mode activates. Reactive fallback is used until then."

echo ""
echo "=========================================================="
echo " Simulation is running!"
echo ""
echo " Watch replayer  : kubectl logs -n research-workload job/trace-replayer -f"
echo " Watch autoscaler: kubectl get jobs -n research-autoscaler -w"
echo " Watch pods      : kubectl get pods -n research-workload -w"
echo "=========================================================="

# ── Phase 7 — Evidence collection instructions ────────────────────────────────
echo ""
echo ">>> Phase 7: Evidence collection"
echo ""
echo " ┌──────────────────────────────────────────────────────────┐"
echo " │  Wait ~2 hours for predictive mode to activate,         │"
echo " │  then run the automated evidence collector:              │"
echo " └──────────────────────────────────────────────────────────┘"
echo ""
echo "  Step 1 — Start port-forwards in separate terminals:"
echo "    kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090 &"
echo "    kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80 &"
echo "    # Grafana: http://localhost:3000  (admin / research2026)"
echo ""
echo "  Step 2 — Run the automated collector (~10 min, fish-compatible):"
echo "    cd $K8S_DIR"
echo "    python3 collect_evidence.py"
echo "    # Outputs: evidence/decisions.csv, evidence/prometheus_*.json,"
echo "    #          evidence/summary_report.txt, etc."
echo ""
echo "  Step 3 — Take 5 Grafana screenshots (see GCP_RESEARCH_GUIDE.md § 8.4):"
echo "    fig1: Namespace (Workloads) → CPU Usage panel (research-workload)"
echo "    fig2: Workload → sample-workload → Current Replicas panel"
echo "    fig3: Workload → load-generator → CPU Usage panel"
echo "    fig4: Cluster → CPU Utilization stats"
echo "    fig5: Explore → Loki → {namespace=\"research-autoscaler\"} |= \"DECISION\""
echo ""
echo "  Step 4 — When done, run cleanup:"
echo "    bash $K8S_DIR/cleanup.sh"
echo ""
echo "=========================================================="
echo " Full research flow:"
echo "   bash cleanup.sh   ← tears down cluster + clears evidence"
echo "   bash restart.sh   ← this script: full setup + start simulation"
echo "   (wait 2 hours)"
echo "   python3 collect_evidence.py  ← automated evidence export"
echo "   (take 5 Grafana screenshots)"
echo "   bash cleanup.sh   ← stop billing when done"
echo "=========================================================="
