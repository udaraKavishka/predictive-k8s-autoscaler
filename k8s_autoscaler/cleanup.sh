#!/usr/bin/env bash
# cleanup.sh — Delete GKE cluster and clear local evidence
# Run BEFORE restart.sh when starting a fresh research session.
#
# Usage:  bash cleanup.sh
#
# What it does:
#   - Deletes the GKE cluster          (stops all VM billing immediately)
#   - Clears local evidence/ directory (removes stale CSV/log files)
#   - Optionally deletes Artifact Registry (default: KEEP — saves ~15 min rebuild)
#
# What it does NOT do:
#   - Delete local model files (.keras, .pkl) — these are your trained artefacts
#   - Delete local trace CSV              — takes time to re-download
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ID=research-autoscaler-2026
REGION=us-central1
ZONE=us-central1-a
CLUSTER=research-autoscaler
K8S_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================================="
echo " Research Autoscaler — Cleanup"
echo " Project : $PROJECT_ID"
echo " Cluster : $CLUSTER ($ZONE)"
echo ""
echo " This WILL:     delete GKE cluster (billing stops)"
echo "                clear evidence/ directory"
echo " This WON'T:    delete Artifact Registry (saves ~15 min rebuild)"
echo "                delete local model .keras / .pkl files"
echo "=========================================================="
echo ""
read -r -p "Are you sure you want to proceed? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
  echo "Cleanup cancelled."
  exit 0
fi
echo ""

# ── Delete GKE cluster ────────────────────────────────────────────────────────
if gcloud container clusters describe "$CLUSTER" --zone "$ZONE" \
     --project "$PROJECT_ID" &>/dev/null; then
  echo ">>> Deleting GKE cluster '$CLUSTER'..."
  gcloud container clusters delete "$CLUSTER" \
    --zone "$ZONE" \
    --project "$PROJECT_ID" \
    --quiet
  echo "    ✓ Cluster deleted — billing stopped."
else
  echo ">>> No cluster named '$CLUSTER' found — skipping"
fi

# ── Clear local evidence/ ─────────────────────────────────────────────────────
if [ -d "$K8S_DIR/evidence" ]; then
  echo ">>> Clearing evidence/ directory..."
  rm -rf "$K8S_DIR/evidence"
  echo "    ✓ evidence/ cleared."
else
  echo ">>> evidence/ directory not found — skipping"
fi

# ── Optional: delete Artifact Registry ───────────────────────────────────────
echo ""
read -r -p "Delete Artifact Registry too? (Rebuilding image takes ~15 min) (yes/no): " del_reg
if [ "$del_reg" = "yes" ]; then
  if gcloud artifacts repositories describe research \
       --location="$REGION" --project "$PROJECT_ID" &>/dev/null; then
    echo ">>> Deleting Artifact Registry 'research'..."
    gcloud artifacts repositories delete research \
      --location="$REGION" \
      --project "$PROJECT_ID" \
      --quiet
    echo "    ✓ Artifact Registry deleted."
  else
    echo ">>> No Artifact Registry found — skipping"
  fi
else
  echo ">>> Artifact Registry kept (image will be reused in next restart)."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================================="
echo " Cleanup complete."
echo ""
echo " Next step: run the full setup script:"
echo "   bash $K8S_DIR/restart.sh"
echo "=========================================================="
