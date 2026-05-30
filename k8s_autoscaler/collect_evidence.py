#!/usr/bin/env python3
"""
Dissertation Evidence Collector
================================
Automated script to collect all GKE simulation evidence for the dissertation.
Run once after the trace replay has been running for at least 2 hours.

Usage:
    cd /home/udara/Documents/Research/k8s_autoscaler
    python3 collect_evidence.py

Requires:
    - kubectl configured and pointing at the research-autoscaler cluster
    - pip install requests  (for Prometheus queries)
    - Prometheus port-forward running on localhost:9090
      (kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090 &)

Output:
    evidence/
        autoscaler_decisions.log   raw logs from all autoscaler jobs
        decisions.jsonl            extracted DECISION JSON lines
        decisions.csv              structured table for dissertation
        scaling_events.txt         kubectl events from research-workload
        pod_status.txt             pod snapshot (both namespaces)
        job_history.txt            all autoscaler job history
        prometheus_cpu.json        CPU usage time-series from Prometheus
        prometheus_replicas.json   replica count time-series from Prometheus
        summary_report.txt         auto-generated stats summary
"""

import csv
import datetime
import json
import os
import subprocess
import sys
import time

# ── Configuration ─────────────────────────────────────────────────────────────

WAIT_SECONDS       = 120          # 2-minute wait between steps (change to 0 to skip)
AUTOSCALER_NS      = "research-autoscaler"
WORKLOAD_NS        = "research-workload"
PROMETHEUS_URL     = "http://localhost:9090"
OUTPUT_DIR         = os.path.join(os.path.dirname(__file__), "evidence")

# Prometheus query time window — last 3 hours covers a full simulation run
PROM_DURATION      = "3h"

# ── Helpers ───────────────────────────────────────────────────────────────────

def ts():
    """Current time as [HH:MM:SS] prefix."""
    return datetime.datetime.now().strftime("[%H:%M:%S]")


def log(msg):
    print(f"{ts()} {msg}", flush=True)


def wait(reason=""):
    if WAIT_SECONDS <= 0:
        return
    msg = f"Waiting {WAIT_SECONDS // 60} min {WAIT_SECONDS % 60}s"
    if reason:
        msg += f" — {reason}"
    log(msg)
    for remaining in range(WAIT_SECONDS, 0, -10):
        print(f"  {remaining}s remaining…", end="\r", flush=True)
        time.sleep(min(10, remaining))
    print(" " * 40, end="\r")  # clear the countdown line


def run(cmd, capture=True):
    """Run a shell command, return stdout string. Prints error but does not abort."""
    result = subprocess.run(cmd, capture_output=capture, text=True)
    if result.returncode != 0 and result.stderr:
        log(f"  ⚠  {' '.join(cmd[:3])} returned code {result.returncode}: {result.stderr.strip()[:120]}")
    return result.stdout or ""


def save(filename, content):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        f.write(content)
    log(f"  ✓  Saved {filename}  ({len(content.splitlines())} lines, {len(content):,} bytes)")
    return path

# ── Steps ─────────────────────────────────────────────────────────────────────

def setup():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    banner = (
        f"{'='*60}\n"
        f"  Dissertation Evidence Collector\n"
        f"  Started: {datetime.datetime.now().isoformat()}\n"
        f"  Output:  {OUTPUT_DIR}\n"
        f"{'='*60}\n"
    )
    print(banner)


def step1_collect_logs():
    log("Step 1/5: Collecting autoscaler job logs…")
    jobs_raw = run(["kubectl", "get", "jobs", "-n", AUTOSCALER_NS, "-o", "name"])
    jobs = [j.strip() for j in jobs_raw.splitlines() if j.strip()]
    log(f"  Found {len(jobs)} jobs in {AUTOSCALER_NS}")

    all_logs = []
    for job in jobs:
        job_name = job.replace("job.batch/", "")
        log_text = run(["kubectl", "logs", "-n", AUTOSCALER_NS, f"job/{job_name}"])
        if log_text:
            all_logs.append(f"# ── {job_name} ──────────────────────────────────────\n")
            all_logs.append(log_text)
            all_logs.append("\n")

    combined = "".join(all_logs)
    save("autoscaler_decisions.log", combined)
    return combined


def step2_extract_decisions(raw_logs):
    log("Step 2/5: Extracting DECISION lines → jsonl + csv…")

    # Extract DECISION JSON lines
    decisions = []
    for line in raw_logs.splitlines():
        if "DECISION:" in line:
            try:
                json_part = line.split("DECISION:", 1)[1].strip()
                decisions.append(json.loads(json_part))
            except json.JSONDecodeError as e:
                log(f"  ⚠  Could not parse decision line: {e}")

    log(f"  Found {len(decisions)} DECISION entries")

    # Save jsonl
    jsonl = "\n".join(json.dumps(d) for d in decisions) + "\n"
    save("decisions.jsonl", jsonl)

    # Save csv
    if decisions:
        csv_path = os.path.join(OUTPUT_DIR, "decisions.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=decisions[0].keys())
            writer.writeheader()
            writer.writerows(decisions)
        log(f"  ✓  Saved decisions.csv  ({len(decisions)} rows)")
    else:
        log("  ⚠  No decisions found — csv not written")

    return decisions


def step3_prometheus():
    log("Step 3/5: Querying Prometheus for time-series data…")
    try:
        import requests
    except ImportError:
        log("  ⚠  requests not installed — skipping Prometheus export")
        log("     Run: pip install requests")
        return

    now = datetime.datetime.utcnow()
    start = (now - datetime.timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end   = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    step  = "300"  # 5-min resolution

    queries = {
        "prometheus_cpu.json": (
            f'avg(rate(container_cpu_usage_seconds_total{{'
            f'namespace="{WORKLOAD_NS}",'
            f'container!="POD",container!=""}}'
            f'[300s])) by (pod)'
        ),
        "prometheus_replicas.json": (
            f'kube_deployment_spec_replicas{{'
            f'namespace="{WORKLOAD_NS}"}}'
        ),
        "prometheus_load_generator_cpu.json": (
            f'avg(rate(container_cpu_usage_seconds_total{{'
            f'namespace="{WORKLOAD_NS}",'
            f'pod=~"load-generator-.*",'
            f'container!="POD",container!=""}}'
            f'[300s]))'
        ),
    }

    for filename, query in queries.items():
        try:
            resp = requests.get(
                f"{PROMETHEUS_URL}/api/v1/query_range",
                params={"query": query, "start": start, "end": end, "step": step},
                timeout=15,
            )
            resp.raise_for_status()
            save(filename, json.dumps(resp.json(), indent=2))
        except Exception as e:
            log(f"  ⚠  Prometheus query failed for {filename}: {e}")
            log(f"     Is port-forward running? kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090 &")


def step4_cluster_state():
    log("Step 4/5: Snapshotting cluster state…")

    # Scaling events
    events = run([
        "kubectl", "get", "events", "-n", WORKLOAD_NS,
        "--sort-by=.lastTimestamp",
        "--field-selector=reason=SuccessfulRescale",
    ])
    # Also get all events if no scaling events found
    if not events.strip():
        events = run([
            "kubectl", "get", "events", "-n", WORKLOAD_NS,
            "--sort-by=.lastTimestamp",
        ])
    save("scaling_events.txt", events)

    # Pod status — both namespaces
    pods_workload  = run(["kubectl", "get", "pods", "-n", WORKLOAD_NS,  "-o", "wide"])
    pods_autoscaler = run(["kubectl", "get", "pods", "-n", AUTOSCALER_NS, "-o", "wide"])
    save("pod_status.txt",
         f"=== {WORKLOAD_NS} ===\n{pods_workload}\n"
         f"=== {AUTOSCALER_NS} ===\n{pods_autoscaler}")

    # Job history
    jobs = run(["kubectl", "get", "jobs", "-n", AUTOSCALER_NS,
                "--sort-by=.metadata.creationTimestamp"])
    save("job_history.txt", jobs)

    # Deployment replica counts
    deploys = run(["kubectl", "get", "deployments", "-n", WORKLOAD_NS, "-o", "wide"])
    save("deployment_status.txt", deploys)


def step5_summary(decisions):
    log("Step 5/5: Generating summary report…")

    if not decisions:
        log("  ⚠  No decisions — summary will be minimal")
        decisions = []

    total        = len(decisions)
    predictive   = sum(1 for d in decisions if d.get("method") == "predictive")
    reactive     = sum(1 for d in decisions if "reactive" in d.get("method", ""))
    replicas_all = [d.get("new_replicas", 1) for d in decisions]
    max_rep      = max(replicas_all) if replicas_all else "—"
    min_rep      = min(replicas_all) if replicas_all else "—"
    scale_ups    = sum(1 for d in decisions
                       if d.get("new_replicas", 0) > d.get("old_replicas", 0))
    scale_downs  = sum(1 for d in decisions
                       if d.get("new_replicas", 0) < d.get("old_replicas", 0))

    method_counts = {}
    for d in decisions:
        m = d.get("method", "unknown")
        method_counts[m] = method_counts.get(m, 0) + 1

    report_lines = [
        "=" * 60,
        "  Kubernetes Simulation — Evidence Summary",
        f"  Generated: {datetime.datetime.now().isoformat()}",
        "=" * 60,
        "",
        f"Total autoscaler decisions   : {total}",
        f"Predictive decisions         : {predictive}  ({predictive/total*100:.1f}%)" if total else "Predictive decisions         : 0",
        f"Reactive fallback decisions  : {reactive}   ({reactive/total*100:.1f}%)" if total else "Reactive fallback decisions  : 0",
        "",
        "Method breakdown:",
        *[f"  {m:<35} {c}" for m, c in sorted(method_counts.items())],
        "",
        f"Max replicas reached         : {max_rep}",
        f"Min replicas reached         : {min_rep}",
        f"Scale-up   events            : {scale_ups}",
        f"Scale-down events            : {scale_downs}",
        "",
        "Output files:",
        *[f"  evidence/{f}" for f in sorted(os.listdir(OUTPUT_DIR))],
        "",
        "=" * 60,
        "  Evidence collection complete — ready for dissertation",
        "=" * 60,
    ]

    report = "\n".join(report_lines)
    save("summary_report.txt", report)
    print("\n" + report)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    setup()

    wait("letting the latest autoscaler CronJob run complete before collecting logs")
    raw_logs  = step1_collect_logs()

    wait("letting Prometheus metrics settle before querying")
    decisions = step2_extract_decisions(raw_logs)

    wait("waiting before Prometheus time-series export")
    step3_prometheus()

    wait("waiting before final cluster snapshot")
    step4_cluster_state()

    # No wait before summary — just parsing already-collected data
    step5_summary(decisions)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{ts()} Interrupted by user — partial evidence saved to {OUTPUT_DIR}/")
        sys.exit(1)
