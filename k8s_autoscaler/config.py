"""
Central configuration for k8s_autoscaler scripts.
Edit the values in the USER SETTINGS section below to change any behaviour.
Environment variables still override these values if set.
"""
import os

# ═══════════════════════════════════════════════════════════════════════════════
#  USER SETTINGS — edit these
# ═══════════════════════════════════════════════════════════════════════════════

# Folder that contains the k8s_autoscaler scripts.
# Change this if you move the directory.
_AUTOSCALER_DIR = "/mnt/30400F81400F4CD2/Uni/Level 04/4.1/Research/Datasets/k8s_autoscaler"

# Where the trained model artifacts live locally
#   hybrid_model_ewc_er.h5 / temporal_scaler.pkl / static_scaler.pkl / target_scaler.pkl
# In Kubernetes this is always /model (PVC mount, set by ConfigMap).
# Change this to point at a different training run folder if needed.
_MODEL_DIR = "/mnt/30400F81400F4CD2/Uni/Level 04/4.1/Research/Datasets/research_imp/linux_gpu_run_v3"

# Full path to the Alibaba timeseries CSV uploaded to the data PVC in Kubernetes.
_DATA_CSV = "/mnt/30400F81400F4CD2/Uni/Level 04/4.1/Research/Datasets/research_imp/linux_gpu_run_v3/alibaba_timeseries_full.csv"

# Where evidence files and figures are written.
_EVIDENCE_DIR = os.path.join(_AUTOSCALER_DIR, "evidence")

# Kubernetes namespaces
_AUTOSCALER_NS = "research-autoscaler"
_WORKLOAD_NS   = "research-workload"

# Prometheus endpoint (local port-forward default)
_PROMETHEUS_URL = "http://localhost:9090"
_PROM_DURATION  = "3h"           # query window for evidence collection

# Scaling parameters — must match k8s/configmap.yaml
_CPU_PER_REPLICA = 0.1
_MIN_REPLICAS    = 1
_MAX_REPLICAS    = 10

# Load replayer — vCPU per stress-ng pod
_CPU_PER_STRESS_POD = 0.1
_MIN_PODS           = 1
_MAX_PODS           = 20

# Seconds to wait between evidence-collection steps (0 = no waits)
_COLLECT_WAIT_SECONDS = 120

# ═══════════════════════════════════════════════════════════════════════════════
#  Resolved values — env vars override the settings above if present
# ═══════════════════════════════════════════════════════════════════════════════

AUTOSCALER_DIR       = os.getenv("AUTOSCALER_DIR",       _AUTOSCALER_DIR)
MODEL_DIR            = os.getenv("MODEL_DIR",             _MODEL_DIR)
DATA_CSV             = os.getenv("DATA_CSV",              _DATA_CSV)
EVIDENCE_DIR         = os.getenv("EVIDENCE_DIR",         _EVIDENCE_DIR)
AUTOSCALER_NS        = os.getenv("AUTOSCALER_NS",        _AUTOSCALER_NS)
WORKLOAD_NS          = os.getenv("WORKLOAD_NS",          _WORKLOAD_NS)
PROMETHEUS_URL       = os.getenv("PROMETHEUS_URL",       _PROMETHEUS_URL)
PROM_DURATION        = os.getenv("PROM_DURATION",        _PROM_DURATION)
CPU_PER_REPLICA      = float(os.getenv("CPU_PER_REPLICA",      str(_CPU_PER_REPLICA)))
MIN_REPLICAS         = int(os.getenv("MIN_REPLICAS",           str(_MIN_REPLICAS)))
MAX_REPLICAS         = int(os.getenv("MAX_REPLICAS",           str(_MAX_REPLICAS)))
CPU_PER_STRESS_POD   = float(os.getenv("STRESS_CPU_PER_POD",  str(_CPU_PER_STRESS_POD)))
MIN_PODS             = int(os.getenv("MIN_PODS",               str(_MIN_PODS)))
MAX_PODS             = int(os.getenv("MAX_PODS",               str(_MAX_PODS)))
COLLECT_WAIT_SECONDS = int(os.getenv("COLLECT_WAIT_SECONDS",  str(_COLLECT_WAIT_SECONDS)))
