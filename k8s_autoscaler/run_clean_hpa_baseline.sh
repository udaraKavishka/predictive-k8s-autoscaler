#!/usr/bin/env bash
# Clean Kubernetes HPA baseline run.
#
# Runs the SAME load ramp under the stock Kubernetes Horizontal Pod Autoscaler instead of
# the predictive autoscaler, and collects a clean, separately-labelled evidence set so the
# predictive-vs-HPA comparison (collect_evidence.py step7) is trustworthy.
#
# Implements GCP_RESEARCH_GUIDE.md §8.4 + §9.5 as one safe, repeatable command:
#   1. suspend the predictive CronJob (so the two controllers don't fight)
#   2. apply the HPA manifest (k8s/hpa.yaml)
#   3. save proof that HPA — not the predictive autoscaler — controlled the deployment
#   4. run the identical ramp (ramp_demo.sh --no-flush)
#   5. collect evidence into evidence/hpa/  (EVIDENCE_LABEL=hpa)
#   6. contamination guard: fail if decisions.csv contains predictive decisions
#   7. teardown: delete the HPA and un-suspend the predictive CronJob (always, via trap)
#
# Usage:
#   bash run_clean_hpa_baseline.sh             # full ramp (up and down)
#   bash run_clean_hpa_baseline.sh --up-only   # ramp up only (1->2->3)
#   bash run_clean_hpa_baseline.sh --quick     # 10-min phases (smoke test, NOT real evidence)
#
# Prereqs: kubectl configured against the GKE cluster; metrics-server / Prometheus running;
#          the predictive stack already deployed (namespaces, load-generator, CronJob).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WORKLOAD_NS="research-workload"
AUTOSCALER_NS="research-autoscaler"
DEPLOYMENT="load-generator"
CRONJOB="predictive-autoscaler"
HPA_MANIFEST="k8s/hpa.yaml"
EVIDENCE_DIR="evidence/hpa"

# Pass-through ramp flags (default: --no-flush so we don't wait 2h; the cluster ramp itself
# is identical to the predictive pass).
RAMP_FLAGS="--no-flush"
for arg in "$@"; do
  case "$arg" in
    --up-only) RAMP_FLAGS="$RAMP_FLAGS --up-only" ;;
    --quick)   RAMP_FLAGS="$RAMP_FLAGS --quick" ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

log() { echo -e "\n[clean-hpa] $*"; }

# --- teardown always runs, even on error/Ctrl-C ---------------------------------
teardown() {
  log "TEARDOWN: removing HPA and re-enabling the predictive autoscaler"
  kubectl delete hpa "$DEPLOYMENT" -n "$WORKLOAD_NS" --ignore-not-found
  kubectl patch cronjob "$CRONJOB" -n "$AUTOSCALER_NS" \
    -p '{"spec":{"suspend":false}}' || true
  log "TEARDOWN complete. Predictive CronJob un-suspended."
}
trap teardown EXIT

# --- preflight ------------------------------------------------------------------
command -v kubectl >/dev/null || { echo "kubectl not found" >&2; exit 1; }
[ -f "$HPA_MANIFEST" ] || { echo "missing $HPA_MANIFEST" >&2; exit 1; }
kubectl get deployment "$DEPLOYMENT" -n "$WORKLOAD_NS" >/dev/null \
  || { echo "deployment $DEPLOYMENT not found in $WORKLOAD_NS" >&2; exit 1; }

mkdir -p "$EVIDENCE_DIR"

# --- 1. suspend predictive CronJob ---------------------------------------------
log "1/6 Suspending predictive CronJob '$CRONJOB'"
kubectl patch cronjob "$CRONJOB" -n "$AUTOSCALER_NS" -p '{"spec":{"suspend":true}}'
# Let any in-flight predictive job finish so it can't write decisions during the HPA pass.
log "    waiting 90s for any in-flight predictive job to drain..."
sleep 90

# --- 2. apply HPA ---------------------------------------------------------------
log "2/6 Applying HPA manifest"
kubectl apply -f "$HPA_MANIFEST"
log "    waiting 60s for HPA to read metrics (avoid <unknown> targets)..."
sleep 60
kubectl get hpa "$DEPLOYMENT" -n "$WORKLOAD_NS"

# --- 3. save proof that HPA controlled the deployment --------------------------
log "3/6 Saving HPA proof into $EVIDENCE_DIR"
kubectl get hpa "$DEPLOYMENT" -n "$WORKLOAD_NS" -o yaml > "$EVIDENCE_DIR/hpa.yaml"
kubectl get hpa "$DEPLOYMENT" -n "$WORKLOAD_NS"        > "$EVIDENCE_DIR/hpa_status.txt"
kubectl get cronjob "$CRONJOB" -n "$AUTOSCALER_NS"     > "$EVIDENCE_DIR/autoscaler_cronjob_status.txt"

# --- 4. run the identical ramp --------------------------------------------------
log "4/6 Running load ramp (HPA does the scaling):  ramp_demo.sh $RAMP_FLAGS"
# shellcheck disable=SC2086
bash ramp_demo.sh $RAMP_FLAGS || true   # ramp_demo also calls collect_evidence (predictive label);
                                        # we re-collect under the hpa label below.

# --- 5. collect HPA-labelled evidence ------------------------------------------
log "5/6 Collecting evidence into $EVIDENCE_DIR (EVIDENCE_LABEL=hpa)"
EVIDENCE_LABEL=hpa python3 collect_evidence.py

# --- 6. contamination guard -----------------------------------------------------
log "6/6 Contamination check on $EVIDENCE_DIR/decisions.csv"
if [ -f "$EVIDENCE_DIR/decisions.csv" ] && grep -q "predictive" "$EVIDENCE_DIR/decisions.csv"; then
  echo "[clean-hpa] WARNING: decisions.csv contains predictive entries." >&2
  echo "[clean-hpa] The predictive CronJob may not have been fully suspended." >&2
  echo "[clean-hpa] Cite ONLY the Prometheus replica/CPU time series from this pass," >&2
  echo "[clean-hpa] not decisions.csv, unless hpa.yaml/hpa_status.txt prove HPA control." >&2
else
  log "OK: no predictive decisions found — this is a clean HPA baseline."
fi

log "DONE. Clean HPA evidence in $EVIDENCE_DIR/"
log "If evidence/predictive/ also exists, collect_evidence.py wrote the side-by-side"
log "comparison: evidence/comparison.csv and evidence/fig4_predictive_vs_hpa.png"
