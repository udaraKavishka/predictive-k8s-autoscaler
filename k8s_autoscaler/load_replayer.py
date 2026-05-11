"""
Alibaba Trace Load Replayer
============================
Research: Adaptive CPU Auto-Scaling Using Hybrid Neural Networks and
          Continual Learning in Kubernetes Orchestration

Reads alibaba_timeseries_full.csv and replays the CPU demand pattern
for a chosen application by scaling a stress-ng Deployment to match
the real trace demand. This creates a realistic non-stationary workload
for the autoscaler to track — providing dissertation-quality evidence.

Usage:
  python load_replayer.py \
    --csv /data/alibaba_timeseries_full.csv \
    --app app_0 \
    --namespace research-workload \
    --deployment load-generator \
    --speed 1.0

Arguments:
  --csv         Path to alibaba_timeseries_full.csv
  --app         app_name to replay (default: first app in file)
  --namespace   Kubernetes namespace
  --deployment  stress-ng Deployment to scale
  --speed       Time multiplier (0.1 = 10× faster for quick demo, 1.0 = real-time)
  --dry-run     Print scaling decisions without applying them
  --log-file    CSV file to log actual vs demanded CPU
"""

import argparse
import csv
import datetime
import json
import logging
import math
import os
import sys
import time

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("load-replayer")

# vCPU allocated per stress-ng pod
CPU_PER_STRESS_POD = float(os.getenv("STRESS_CPU_PER_POD", "0.1"))
MIN_PODS = 1
MAX_PODS = 20

# ---------------------------------------------------------------------------
# Kubernetes helper
# ---------------------------------------------------------------------------

def get_k8s_apps_client():
    try:
        from kubernetes import client as k8s_client, config as k8s_config  # type: ignore
        try:
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            k8s_config.load_kube_config()
        return k8s_client.AppsV1Api()
    except Exception as exc:
        log.error(f"Cannot init Kubernetes client: {exc}")
        return None


def patch_replicas(apps_api, namespace: str, deployment: str, replicas: int, dry_run: bool):
    replicas = max(MIN_PODS, min(MAX_PODS, replicas))
    if dry_run:
        log.info(f"[DRY RUN] Would set {namespace}/{deployment} → {replicas} replicas")
        return
    try:
        apps_api.patch_namespaced_deployment_scale(
            name=deployment,
            namespace=namespace,
            body={"spec": {"replicas": replicas}},
        )
        log.info(f"Set {namespace}/{deployment} → {replicas} replicas")
    except Exception as exc:
        log.error(f"Patch failed: {exc}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trace(csv_path: str, app_name: str | None) -> pd.DataFrame:
    log.info(f"Loading trace from {csv_path}")
    df = pd.read_csv(csv_path)

    required = {"app_name", "timestamp", "cpu_demand"}
    missing = required - set(df.columns)
    if missing:
        log.error(f"CSV missing columns: {missing}")
        sys.exit(1)

    if app_name is None:
        app_name = df["app_name"].iloc[0]
        log.info(f"No app specified, using first app: {app_name}")

    app_df = df[df["app_name"] == app_name].copy()
    if app_df.empty:
        available = df["app_name"].unique()[:10].tolist()
        log.error(f"App '{app_name}' not found. Available (first 10): {available}")
        sys.exit(1)

    app_df = app_df.sort_values("timestamp").reset_index(drop=True)
    log.info(f"Loaded {len(app_df)} timesteps for app '{app_name}'")
    log.info(f"  CPU demand range: {app_df['cpu_demand'].min():.2f} – {app_df['cpu_demand'].max():.2f} vCPUs")
    return app_df


# ---------------------------------------------------------------------------
# Main replay loop
# ---------------------------------------------------------------------------

def replay(args):
    df = load_trace(args.csv, args.app)
    apps_api = None if args.dry_run else get_k8s_apps_client()

    interval_real = 300.0 / args.speed  # wall-clock seconds between steps
    log.info(
        f"Replaying {len(df)} steps at {args.speed}× speed "
        f"({interval_real:.1f}s per step)"
    )

    log_rows = []
    start_wall = time.monotonic()

    for idx, row in df.iterrows():
        step_start = time.monotonic()
        cpu_demand = float(row["cpu_demand"])
        timestamp_sim = row["timestamp"]

        # Number of stress-ng pods needed to represent this CPU demand
        desired_pods = max(MIN_PODS, math.ceil(cpu_demand / CPU_PER_STRESS_POD))
        desired_pods = min(MAX_PODS, desired_pods)

        log.info(
            f"Step {idx+1:4d}/{len(df)} | sim_t={timestamp_sim} | "
            f"cpu_demand={cpu_demand:.3f} vCPU | desired_pods={desired_pods}"
        )

        if apps_api or args.dry_run:
            patch_replicas(apps_api, args.namespace, args.deployment, desired_pods, args.dry_run)

        log_rows.append({
            "step":         idx + 1,
            "sim_timestamp": timestamp_sim,
            "wall_time":    datetime.datetime.utcnow().isoformat() + "Z",
            "cpu_demand":   cpu_demand,
            "desired_pods": desired_pods,
        })

        # Sleep until next step
        elapsed = time.monotonic() - step_start
        sleep_time = max(0.0, interval_real - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Write log
    if args.log_file:
        log.info(f"Writing replay log to {args.log_file}")
        pd.DataFrame(log_rows).to_csv(args.log_file, index=False)

    total_wall = time.monotonic() - start_wall
    log.info(
        f"Replay complete: {len(df)} steps in {total_wall/60:.1f} minutes "
        f"(simulated {len(df)*5} minutes)"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Replay Alibaba trace as Kubernetes load")
    p.add_argument("--csv",        required=True, help="Path to alibaba_timeseries_full.csv")
    p.add_argument("--app",        default=None,  help="app_name to replay (default: first app)")
    p.add_argument("--namespace",  default="research-workload")
    p.add_argument("--deployment", default="load-generator")
    p.add_argument("--speed",      type=float, default=1.0,
                   help="Time multiplier (e.g. 10.0 = 10× faster)")
    p.add_argument("--dry-run",    action="store_true")
    p.add_argument("--log-file",   default="replay_log.csv",
                   help="Output CSV for replay evidence")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    replay(args)
