"""
Predictive CPU Auto-Scaler for Kubernetes
==========================================
Research: Adaptive CPU Auto-Scaling Using Hybrid Neural Networks and
          Continual Learning in Kubernetes Orchestration

Uses the trained hybrid LSTM+MLP model (hybrid_model_ewc_er.keras) to predict
CPU demand exactly 30 minutes ahead and scales a target Deployment accordingly.

Model I/O (must match training in research_imp_V2.py):
  Input 1 - temporal:  (1, 24, 10)   24 x 5-min steps, 10 features
  Input 2 - static:    (1, 7)         6 workload features + app_id
  Output  - scalar:    (1, 1)         predicted CPU (scaled); invert with target_scaler

Required files (produced by research_imp_V2.py, upload to /model/ PVC):
  hybrid_model_ewc_er.keras   trained model
  temporal_scaler.pkl          StandardScaler for temporal features
  static_scaler.pkl            StandardScaler for static features
  target_scaler.pkl            StandardScaler for target (cpu_demand)

Note on pickle: The .pkl scaler files are loaded with pickle because research_imp_V2.py
saved them with pickle.dump(). These are your own trusted files produced by training.

Environment variables:
  MODEL_DIR           Directory containing .keras + .pkl files  (default: /model)
  PROMETHEUS_URL      Prometheus base URL   (default: http://prometheus-operated:9090)
  TARGET_NAMESPACE    K8s namespace         (default: research-workload)
  TARGET_DEPLOYMENT   Deployment to scale   (default: sample-workload)
  CPU_PER_REPLICA     vCPU per replica      (default: 0.1)
  MIN_REPLICAS        Minimum replicas      (default: 1)
  MAX_REPLICAS        Maximum replicas      (default: 10)
  SLA_TOLERANCE       Over-prov margin      (default: 0.15)
  APP_ID              Numeric app identifier for static input (default: 0)
  DRY_RUN             If "true", log only, no K8s patch  (default: false)
"""

import datetime
import json
import logging
import math
import os
import pickle  
import sys
import tempfile
import zipfile
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("autoscaler")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR         = os.getenv("MODEL_DIR",          "/model")
PROMETHEUS_URL    = os.getenv("PROMETHEUS_URL",      "http://prometheus-operated:9090")
TARGET_NAMESPACE  = os.getenv("TARGET_NAMESPACE",    "research-workload")
TARGET_DEPLOYMENT = os.getenv("TARGET_DEPLOYMENT",   "sample-workload")
CPU_PER_REPLICA   = float(os.getenv("CPU_PER_REPLICA",  "0.1"))
MIN_REPLICAS      = int(os.getenv("MIN_REPLICAS",    "1"))
MAX_REPLICAS      = int(os.getenv("MAX_REPLICAS",    "10"))
SLA_TOLERANCE     = float(os.getenv("SLA_TOLERANCE", "0.15"))
APP_ID            = float(os.getenv("APP_ID",        "0"))
DRY_RUN           = os.getenv("DRY_RUN", "false").lower() == "true"

HISTORY_LENGTH = 24   # 24 x 5-min = 2-hour lookback (fixed in training)
INTERVAL_SEC   = 300  # 5 minutes
ROLL_WINDOW    = 12   # 1-hour rolling window (matches cfg.roll_window in training)


# ---------------------------------------------------------------------------
# Load model + scalers
# ---------------------------------------------------------------------------
def _build_model(keras):
    """Reconstruct model architecture — mirrors create_hybrid_model in research_imp_V3.py."""
    from keras.layers import Input, LSTM, Dense, BatchNormalization, Dropout, concatenate
    from keras.models import Model

    lstm_input = Input(shape=(HISTORY_LENGTH, 10), name="temporal_input")
    x = LSTM(128, return_sequences=True, dropout=0.2, name="lstm_128")(lstm_input)
    x = LSTM(64,  return_sequences=True, dropout=0.2, name="lstm_64")(x)
    x = LSTM(32, name="lstm_32")(x)
    lstm_embed = Dense(16, activation="relu", name="lstm_embedding")(x)

    mlp_input = Input(shape=(7,), name="static_input")
    x = Dense(64, activation="relu", name="mlp_64")(mlp_input)
    x = BatchNormalization(name="mlp_bn")(x)
    x = Dropout(0.2, name="mlp_drop")(x)
    x = Dense(32, activation="relu", name="mlp_32")(x)
    mlp_embed = Dense(16, activation="relu", name="mlp_embedding")(x)

    fused = concatenate([lstm_embed, mlp_embed], name="fusion")
    fused = Dense(16, activation="relu", name="fusion_dense")(fused)
    output = Dense(1, activation="linear", name="cpu_forecast")(fused)

    return Model(inputs=[lstm_input, mlp_input], outputs=output)


def load_artifacts(model_dir: str):
    """
    Returns (model, temporal_scaler, static_scaler, target_scaler).
    All four files must exist in model_dir — they are produced by research_imp_V2.py.
    """
    import keras  
    model_path    = os.path.join(model_dir, "hybrid_model_ewc_er.keras")
    t_scaler_path = os.path.join(model_dir, "temporal_scaler.pkl")
    s_scaler_path = os.path.join(model_dir, "static_scaler.pkl")
    y_scaler_path = os.path.join(model_dir, "target_scaler.pkl")

    for path in [model_path, t_scaler_path, s_scaler_path, y_scaler_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required artifact missing: {path}")

    log.info(f"Loading model from {model_path}")
    model = _build_model(keras)
    with zipfile.ZipFile(model_path, "r") as zf:
        with tempfile.TemporaryDirectory() as tmp:
            zf.extract("model.weights.h5", tmp)
            model.load_weights(os.path.join(tmp, "model.weights.h5"))
    log.info("Model weights loaded successfully")


    with open(t_scaler_path, "rb") as fh:
        temporal_scaler = pickle.load(fh)  
    with open(s_scaler_path, "rb") as fh:
        static_scaler = pickle.load(fh)  
    with open(y_scaler_path, "rb") as fh:
        target_scaler = pickle.load(fh)  

    log.info("Model and all three scalers loaded")
    return model, temporal_scaler, static_scaler, target_scaler


# ---------------------------------------------------------------------------
# Feature engineering
# Mirrors the exact feature construction in research_imp_V2.py (lines 1590-1609)
#
# Temporal features (10 columns, order must match temporal_scaler):
#   0  cpu_demand
#   1  cpu_diff
#   2  cpu_roll_mean   (12-step = 1-hr rolling mean)
#   3  cpu_roll_std
#   4  cpu_roll_min
#   5  cpu_roll_max
#   6  hour_sin
#   7  hour_cos
#   8  dow_sin
#   9  dow_cos
#
# Static features (7 columns, order must match static_scaler):
#   0  gpu_request_mean
#   1  memory_request_mean
#   2  rdma_request_mean
#   3  role_hn_fraction
#   4  instance_count
#   5  max_instance_per_node
#   6  app_id
# ---------------------------------------------------------------------------

def build_temporal_matrix(cpu_history: np.ndarray) -> np.ndarray:
    """
    Build (HISTORY_LENGTH, 10) temporal feature matrix from raw CPU history.
    cpu_history: 1-D array of HISTORY_LENGTH raw vCPU values (not scaled).
    """
    n = len(cpu_history)
    feats = np.zeros((n, 10), dtype=np.float32)

    feats[:, 0] = cpu_history

    feats[:, 1] = np.diff(cpu_history, prepend=cpu_history[0])

    for i in range(n):
        w = cpu_history[max(0, i - ROLL_WINDOW + 1): i + 1]
        feats[i, 2] = w.mean()
        feats[i, 3] = w.std() if len(w) > 1 else 0.0
        feats[i, 4] = w.min()
        feats[i, 5] = w.max()

    now_utc = datetime.datetime.utcnow()
    for i in range(n):
        step_time = now_utc - datetime.timedelta(seconds=(n - 1 - i) * INTERVAL_SEC)
        hour_frac = step_time.hour + step_time.minute / 60.0
        dow       = step_time.weekday()
        feats[i, 6] = math.sin(2 * math.pi * hour_frac / 24.0)
        feats[i, 7] = math.cos(2 * math.pi * hour_frac / 24.0)
        feats[i, 8] = math.sin(2 * math.pi * dow / 7.0)
        feats[i, 9] = math.cos(2 * math.pi * dow / 7.0)

    return feats


def build_static_vector() -> np.ndarray:
    """
    Build (7,) static feature vector from env vars.
    Represents the workload's resource profile — same features as training.
    """
    return np.array([
        float(os.getenv("STATIC_GPU_REQUEST",    "0.0")),
        float(os.getenv("STATIC_MEM_GIB",        "0.128")),
        float(os.getenv("STATIC_RDMA",           "0.0")),
        float(os.getenv("STATIC_HN_FRACTION",    "0.0")),
        float(os.getenv("STATIC_INSTANCE_COUNT", "1.0")),
        float(os.getenv("STATIC_MAX_PER_NODE",   "0.0")),
        APP_ID,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Prometheus
# ---------------------------------------------------------------------------
def fetch_cpu_history(namespace: str, deployment: str) -> Optional[np.ndarray]:
    """
    Query Prometheus for the last HISTORY_LENGTH x 5-min average CPU samples.
    Returns raw vCPU values (not scaled), shape (HISTORY_LENGTH,), or None.
    """
    try:
        import requests  
    except ImportError:
        log.error("requests library not installed")
        return None

    now   = datetime.datetime.utcnow()
    start = now - datetime.timedelta(seconds=(HISTORY_LENGTH + 2) * INTERVAL_SEC)

    query = (
        f'avg(rate(container_cpu_usage_seconds_total{{'
        f'namespace="{namespace}",'
        f'pod=~"{deployment}-.*",'
        f'container!="POD",container!=""}}'
        f'[{INTERVAL_SEC}s]))'
    )

    try:
        import requests  
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params={
                "query": query,
                "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end":   now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "step":  f"{INTERVAL_SEC}s",
            },
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("data", {}).get("result", [])
        if not results:
            log.warning("Prometheus returned no results for CPU query")
            return None

        values = [float(v[1]) for v in results[0]["values"]]
        if len(values) < HISTORY_LENGTH:
            log.warning(
                f"Only {len(values)} Prometheus samples, need {HISTORY_LENGTH} "
                f"— insufficient history for model; will use reactive fallback"
            )
            # Do NOT pad with zeros — zero-padding corrupts the LSTM temporal
            # features and causes wildly wrong predictions.  Return None so the
            # caller falls back to reactive scaling until we have enough history.
            return None

        return np.array(values[-HISTORY_LENGTH:], dtype=np.float32)

    except Exception as exc:
        log.warning(f"Prometheus query failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Kubernetes
# ---------------------------------------------------------------------------
def get_k8s_apps_client():
    try:
        from kubernetes import client as k8s_client, config as k8s_config  # type: ignore
        try:
            k8s_config.load_incluster_config()
            log.info("Kubernetes: using in-cluster config")
        except k8s_config.ConfigException:
            k8s_config.load_kube_config()
            log.info("Kubernetes: using kubeconfig file")
        return k8s_client.AppsV1Api()
    except Exception as exc:
        log.error(f"Kubernetes client init failed: {exc}")
        return None


def get_current_replicas(apps_api, namespace: str, deployment: str) -> Optional[int]:
    try:
        dep = apps_api.read_namespaced_deployment(deployment, namespace)
        return dep.spec.replicas or 1
    except Exception as exc:
        log.error(f"Cannot read Deployment {namespace}/{deployment}: {exc}")
        return None


def patch_replicas(apps_api, namespace: str, deployment: str, replicas: int) -> bool:
    if DRY_RUN:
        log.info(f"[DRY RUN] Would patch {namespace}/{deployment} -> {replicas} replicas")
        return True
    try:
        apps_api.patch_namespaced_deployment_scale(
            name=deployment,
            namespace=namespace,
            body={"spec": {"replicas": replicas}},
        )
        log.info(f"Patched {namespace}/{deployment} -> {replicas} replicas")
        return True
    except Exception as exc:
        log.error(f"Failed to patch Deployment: {exc}")
        return False


# ---------------------------------------------------------------------------
# Reactive fallback (mirrors HPA-Reactive baseline from research)
# ---------------------------------------------------------------------------
def reactive_scale(current_cpu: float) -> int:
    desired = math.ceil(current_cpu * (1.0 + SLA_TOLERANCE) / CPU_PER_REPLICA)
    return max(MIN_REPLICAS, min(MAX_REPLICAS, desired))


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------
def predict_cpu_demand(
    model,
    temporal_scaler,
    static_scaler,
    target_scaler,
    cpu_history: np.ndarray,
) -> float:
    """
    Run the hybrid LSTM+MLP model and return predicted CPU (vCPU) at t+30min.

    Pipeline (mirrors inference path from research_imp_V2.py):
      1. Build raw temporal matrix (24, 10)
      2. Scale with temporal_scaler
      3. Build raw static vector (7,)
      4. Scale with static_scaler
      5. model.predict -> scaled output (1, 1)
      6. inverse_transform with target_scaler -> raw vCPU
    """
    temporal_raw = build_temporal_matrix(cpu_history)        # (24, 10)
    static_raw   = build_static_vector()                     # (7,)

    temporal_scaled = temporal_scaler.transform(temporal_raw)               # (24, 10)
    static_scaled   = static_scaler.transform(static_raw.reshape(1, -1))   # (1, 7)

    t_input = temporal_scaled[np.newaxis, :, :]   # (1, 24, 10)
    s_input = static_scaled                        # (1, 7)

    pred_scaled = model.predict([t_input, s_input], verbose=0)   # (1, 1)
    pred_cpu    = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0])

    return max(0.0, pred_cpu)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info("Predictive Auto-Scaler starting")
    log.info(f"  Target:     {TARGET_NAMESPACE}/{TARGET_DEPLOYMENT}")
    log.info(f"  Model dir:  {MODEL_DIR}")
    log.info(f"  Prometheus: {PROMETHEUS_URL}")
    log.info(f"  Dry-run:    {DRY_RUN}")
    log.info("=" * 60)

    timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    # Load model and scalers
    try:
        model, temporal_scaler, static_scaler, target_scaler = load_artifacts(MODEL_DIR)
    except Exception as exc:
        log.error(f"Cannot load artifacts: {exc}")
        model = temporal_scaler = static_scaler = target_scaler = None

    # Init Kubernetes client
    apps_api = get_k8s_apps_client()
    current_replicas = 1
    if apps_api:
        r = get_current_replicas(apps_api, TARGET_NAMESPACE, TARGET_DEPLOYMENT)
        if r is not None:
            current_replicas = r

    # Fetch CPU history from Prometheus
    cpu_history = fetch_cpu_history(TARGET_NAMESPACE, TARGET_DEPLOYMENT)

    # Make decision
    if cpu_history is not None and model is not None:
        try:
            predicted_cpu    = predict_cpu_demand(model, temporal_scaler, static_scaler,
                                                  target_scaler, cpu_history)

            # Sanity check: sklearn version mismatch (scalers saved with 1.8.0, loaded
            # with 1.4.2) can cause inverse_transform to return values hundreds of times
            # larger than reality.  Cap at 50× the observed current CPU; anything above
            # that is a scaler artifact, not a real prediction.
            current_cpu_val = float(cpu_history[-1])
            sanity_cap = max(1.0, current_cpu_val * 50)
            if predicted_cpu > sanity_cap:
                log.warning(
                    f"Prediction {predicted_cpu:.2f} vCPU exceeds sanity cap "
                    f"{sanity_cap:.2f} (50× current {current_cpu_val:.4f}) — "
                    f"likely sklearn version mismatch; falling back to reactive"
                )
                predicted_cpu = current_cpu_val
                desired       = reactive_scale(predicted_cpu)
                method        = "reactive-fallback-sanity"
            else:
                provisioned_cpu  = predicted_cpu * (1.0 + SLA_TOLERANCE)
                desired          = max(MIN_REPLICAS, min(MAX_REPLICAS,
                                       math.ceil(provisioned_cpu / CPU_PER_REPLICA)))
                method           = "predictive"
                log.info(
                    f"Predicted t+30min CPU: {predicted_cpu:.4f} vCPU "
                    f"(+{SLA_TOLERANCE*100:.0f}% -> {provisioned_cpu:.4f}) "
                    f"-> {desired} replicas"
                )
        except Exception as exc:
            log.warning(f"Prediction failed ({exc}) -- falling back to reactive")
            predicted_cpu = float(cpu_history[-1])
            desired       = reactive_scale(predicted_cpu)
            method        = "reactive-fallback"
    elif cpu_history is not None:
        log.warning("No model -- using reactive fallback")
        predicted_cpu = float(cpu_history[-1])
        desired       = reactive_scale(predicted_cpu)
        method        = "reactive-fallback"
    else:
        log.warning("No CPU history available -- cannot scale")
        sys.exit(0)

    # Log decision as JSON line (for dissertation evidence CSV)
    decision = {
        "timestamp":     timestamp,
        "current_cpu":   float(cpu_history[-1]) if cpu_history is not None else None,
        "predicted_cpu": predicted_cpu,
        "old_replicas":  current_replicas,
        "new_replicas":  desired,
        "method":        method,
    }
    log.info(f"DECISION: {json.dumps(decision)}")

    # Apply
    if desired != current_replicas:
        if apps_api:
            patch_replicas(apps_api, TARGET_NAMESPACE, TARGET_DEPLOYMENT, desired)
        else:
            log.error("No Kubernetes client -- cannot apply scaling decision")
    else:
        log.info(f"No change needed (replicas stay at {current_replicas})")

    log.info("Auto-scaler run complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.critical("Unhandled exception — autoscaler crashed", exc_info=True)
        sys.exit(1)
