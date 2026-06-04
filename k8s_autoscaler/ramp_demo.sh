#!/usr/bin/env bash
# ramp_demo.sh — Drive load-generator through CPU phases to produce
# 1→2→3 (and 3→2→1) replica scaling evidence from the predictive autoscaler.
#
# Strategy: change stress-ng --cpu-load % to vary avg() CPU per pod.
# The autoscaler reads avg() from Prometheus, predicts 30 min ahead,
# and patches the deployment — producing replica transitions in decisions.jsonl.
#
# CPU thresholds (CPU_PER_REPLICA=0.1, SLA_TOLERANCE=0.15):
#   --cpu-load 25% → avg ~0.050 vCPU → desired = 1 replica
#   --cpu-load 50% → avg ~0.100 vCPU → desired = 2 replicas
#   --cpu-load 80% → avg ~0.160 vCPU → desired = 3 replicas
#
# Usage:
#   bash ramp_demo.sh             # full demo: flush + ramp up + ramp down (~4 h)
#   bash ramp_demo.sh --no-flush  # skip 2-hr flush, start ramp immediately
#   bash ramp_demo.sh --up-only   # ramp up only (1→2→3), no scale-down phase
#   bash ramp_demo.sh --quick     # 10-min phases (test script logic, not real evidence)

set -euo pipefail

NS="research-workload"
DEP="load-generator"
AUTOSCALER_NS="research-autoscaler"
EVIDENCE_DIR="$(dirname "$0")/evidence"

# Phase durations (seconds). Default = 25 min = 5 autoscaler cycles.
# Use --quick to override to 10 min for rapid testing.
PHASE_WAIT=1500   # 25 min
FLUSH_WAIT=7200   # 2 hr — full Prometheus history flush (24 × 5-min steps)

NO_FLUSH=false
UP_ONLY=false

for arg in "$@"; do
  case $arg in
    --no-flush) NO_FLUSH=true ;;
    --up-only)  UP_ONLY=true  ;;
    --quick)    PHASE_WAIT=600; FLUSH_WAIT=600 ;;
  esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

log() { echo "[$(ts)] $*"; }

set_cpu_load() {
  local pct=$1
  log "Setting load-generator --cpu-load ${pct}% ..."
  kubectl patch deployment "$DEP" -n "$NS" \
    -p "{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"stressor\",\"args\":[\"--cpu\",\"1\",\"--cpu-load\",\"${pct}\",\"--timeout\",\"0\"]}]}}}}"
  kubectl rollout status deployment/"$DEP" -n "$NS" --timeout=120s
  log "  → deployment rolled out at cpu-load ${pct}%"
}

show_status() {
  log "Current autoscaler decisions (last 3):"
  tail -3 "$EVIDENCE_DIR/decisions.jsonl" 2>/dev/null | python3 -c \
    "import sys,json; [print(f'  replicas={r[\"old_replicas\"]}→{r[\"new_replicas\"]} cpu={r[\"current_cpu\"]:.4f} pred={r[\"predicted_cpu\"]:.4f} method={r[\"method\"]}') for r in (json.loads(l) for l in sys.stdin)]" \
    2>/dev/null || echo "  (no decisions yet in evidence/decisions.jsonl)"
  echo ""
}

wait_phase() {
  local label=$1
  local seconds=$2
  local minutes=$((seconds / 60))
  log "Waiting ${minutes} min for Prometheus history to reflect '${label}' ..."
  log "  (autoscaler fires every 5 min — expect ${minutes}/5 = $((minutes / 5)) decisions)"
  sleep "$seconds"
}

# ── Pre-flight ────────────────────────────────────────────────────────────────

log "============================================================"
log " Predictive Autoscaler — Replica Ramp Demo"
log " Target    : ${NS}/${DEP}"
log " No-flush  : ${NO_FLUSH}"
log " Up-only   : ${UP_ONLY}"
log " Phase wait: $((PHASE_WAIT / 60)) min"
log "============================================================"
echo ""

# Verify kubectl access
kubectl get deployment "$DEP" -n "$NS" --no-headers 2>/dev/null \
  || { log "ERROR: cannot reach deployment ${NS}/${DEP}. Check kubectl context."; exit 1; }

CURRENT_LOAD=$(kubectl get deployment "$DEP" -n "$NS" \
  -o jsonpath='{.spec.template.spec.containers[0].args}' 2>/dev/null || echo "unknown")
log "Current stress-ng args: ${CURRENT_LOAD}"
show_status

# ── Phase 0: FLUSH (optional) ────────────────────────────────────────────────
# Drop to low CPU so the 2-hour Prometheus history flushes before ramping.
# Skip with --no-flush if the cluster has been idle or was already at low load.

if [ "$NO_FLUSH" = false ]; then
  log "──────────────────────────────────────────"
  log " PHASE 0: FLUSH — drop to cpu-load 25%"
  log " Wait ${FLUSH_WAIT}s = $((FLUSH_WAIT / 60)) min for history to flush"
  log " (You can Ctrl-C and re-run with --no-flush to skip this later)"
  log "──────────────────────────────────────────"
  set_cpu_load 25
  wait_phase "flush (cpu-load 25%)" "$FLUSH_WAIT"
  show_status
fi

# ── Phase 1: BASELINE at 1 replica ───────────────────────────────────────────

log "──────────────────────────────────────────"
log " PHASE 1: BASELINE — cpu-load 25% → expected 1 replica"
log "──────────────────────────────────────────"
set_cpu_load 25
wait_phase "baseline (cpu-load 25%)" "$PHASE_WAIT"
show_status

# ── Phase 2: RAMP UP → 2 replicas ────────────────────────────────────────────

log "──────────────────────────────────────────"
log " PHASE 2: RAMP UP — cpu-load 50% → expected 2 replicas"
log "──────────────────────────────────────────"
set_cpu_load 50
wait_phase "ramp-up medium (cpu-load 50%)" "$PHASE_WAIT"
show_status

# ── Phase 3: RAMP UP → 3 replicas ────────────────────────────────────────────

log "──────────────────────────────────────────"
log " PHASE 3: RAMP UP — cpu-load 80% → expected 3 replicas"
log "──────────────────────────────────────────"
set_cpu_load 80
wait_phase "ramp-up high (cpu-load 80%)" "$PHASE_WAIT"
show_status

# ── Phase 4 & 5: RAMP DOWN (optional) ────────────────────────────────────────

if [ "$UP_ONLY" = false ]; then
  log "──────────────────────────────────────────"
  log " PHASE 4: RAMP DOWN — cpu-load 50% → expected 2 replicas"
  log "──────────────────────────────────────────"
  set_cpu_load 50
  wait_phase "ramp-down medium (cpu-load 50%)" "$PHASE_WAIT"
  show_status

  log "──────────────────────────────────────────"
  log " PHASE 5: RAMP DOWN — cpu-load 25% → expected 1 replica"
  log "──────────────────────────────────────────"
  set_cpu_load 25
  wait_phase "ramp-down baseline (cpu-load 25%)" "$PHASE_WAIT"
  show_status
fi

# ── Restore to original load ──────────────────────────────────────────────────

log "Restoring load-generator to original cpu-load 80% ..."
set_cpu_load 80

# ── Summary ───────────────────────────────────────────────────────────────────

log "============================================================"
log " Demo complete. Running evidence collector..."
log "============================================================"

show_status

cd "$(dirname "$0")"
python3 collect_evidence.py

log "============================================================"
log " Evidence collection done."
log ""
log " Replica transition decisions to look for:"
log "   old_replicas=1, new_replicas=2  (scale-up: 1→2)"
log "   old_replicas=2, new_replicas=3  (scale-up: 2→3)"
log "   old_replicas=3, new_replicas=2  (scale-down: 3→2)"
log "   old_replicas=2, new_replicas=1  (scale-down: 2→1)"
log "============================================================"
