#!/usr/bin/env python3
"""
Dissertation Evidence Collector
================================
Automated script to collect all GKE simulation evidence for the dissertation.
Run once after the trace replay has been running for at least 2 hours.

Usage:
    cd /home/udara/Documents/Research/k8s_autoscaler

    # Single (predictive) run — writes into evidence/
    python3 collect_evidence.py

    # Two-pass predictive-vs-HPA comparison — write into labelled sub-folders:
    EVIDENCE_LABEL=predictive python3 collect_evidence.py
    EVIDENCE_LABEL=hpa        python3 collect_evidence.py   # (or --label hpa)
    # When both evidence/predictive/ and evidence/hpa/ exist, the second run also
    # emits evidence/comparison.csv + evidence/fig4_predictive_vs_hpa.png.

Requires:
    - kubectl configured and pointing at the research-autoscaler cluster
    - pip install requests  (for Prometheus queries)
    - pip install matplotlib (for the figure step; skipped gracefully if missing)
    - Prometheus port-forward running on localhost:9090
      (kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090 &)

Output (in evidence/ or evidence/<EVIDENCE_LABEL>/):
    autoscaler_decisions.log   raw logs from all autoscaler jobs
    decisions.jsonl            extracted DECISION JSON lines
    decisions.csv              structured table for dissertation
    scaling_events.txt         kubectl events from research-workload
    pod_status.txt             pod snapshot (both namespaces)
    job_history.txt            all autoscaler job history
    deployment_status.txt      deployment replica counts
    prometheus_cpu.json        per-pod CPU usage time-series from Prometheus
    prometheus_load_generator_cpu.json  avg load-generator CPU time-series
    prometheus_replicas.json   replica count time-series from Prometheus
    summary_report.txt         auto-generated stats summary
    fig1_cpu_demand_curve.png  load-generator CPU vs time (with scaling thresholds)
    fig2_replica_count.png     replica count step plot (the scaling response)
    fig3_autoscaler_decisions.png  current vs predicted CPU + replicas (predictive only)

Comparison output (only when both labelled passes exist, written to evidence/):
    comparison.csv             predictive vs HPA: SLA-violation % and over-provision %
    fig4_predictive_vs_hpa.png grouped bar chart of the two methods
"""

import argparse
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

# Base evidence dir; a label (env EVIDENCE_LABEL / --label) writes into a sub-folder
# e.g. evidence/predictive/ and evidence/hpa/ for the comparison passes.
BASE_EVIDENCE_DIR  = os.path.join(os.path.dirname(__file__), "evidence")
EVIDENCE_LABEL     = ""            # set in main() from CLI / env
OUTPUT_DIR         = BASE_EVIDENCE_DIR

# vCPU allocated per replica of the target Deployment — must match
# CPU_PER_REPLICA in k8s/configmap.yaml. Used for the figures and the comparison.
CPU_PER_REPLICA    = 0.1

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
    log("Step 1/5: Collecting autoscaler logs…")
    # Pull logs from the pods directly (label set by the CronJob jobTemplate) rather than
    # `kubectl logs job/<name>`. A Completed pod returns its logs instantly; pods that were
    # garbage-collected simply aren't listed — so no ~20s "timed out waiting for the
    # condition" waits and no job-deletion race ("not found"), and every available DECISION
    # line is captured.
    pods_raw = run(["kubectl", "get", "pods", "-n", AUTOSCALER_NS,
                    "-l", "app=predictive-autoscaler", "-o", "name"])
    pods = [p.strip() for p in pods_raw.splitlines() if p.strip()]
    log(f"  Found {len(pods)} autoscaler pods in {AUTOSCALER_NS}")

    all_logs = []
    for pod in pods:
        pod_name = pod.replace("pod/", "")
        log_text = run(["kubectl", "logs", "-n", AUTOSCALER_NS, pod])
        if log_text:
            all_logs.append(f"# ── {pod_name} ──────────────────────────────────────\n")
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

    now = datetime.datetime.now(datetime.timezone.utc)
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

    warnings = []
    if EVIDENCE_LABEL.lower() == "hpa" and predictive:
        warnings.extend([
            "WARNING: This run is labelled 'hpa' but contains predictive-autoscaler decisions.",
            "Do not cite decisions.csv as HPA decisions unless saved HPA status proves HPA controlled the deployment.",
        ])

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
        *warnings,
        "" if warnings else None,
        "Output files:",
        *[f"  evidence/{f}" for f in sorted(os.listdir(OUTPUT_DIR))],
        "",
        "=" * 60,
        "  Evidence collection complete — ready for dissertation",
        "=" * 60,
    ]

    report = "\n".join(line for line in report_lines if line is not None)
    save("summary_report.txt", report)
    print("\n" + report)


# ── Figure / comparison helpers ───────────────────────────────────────────────

def _read_json(filename, base=None):
    """Load a JSON file from OUTPUT_DIR (or `base`); return None if missing/invalid."""
    path = os.path.join(base or OUTPUT_DIR, filename)
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _matrix_series(doc, match=None):
    """Extract (timestamps, values) from a Prometheus query_range matrix.

    `match` is an optional dict of metric-label filters (e.g. {"deployment": "load-generator"}).
    Returns ([float ts], [float val]) for the first matching series, or ([], []).
    """
    if not doc or doc.get("data", {}).get("resultType") != "matrix":
        return [], []
    for series in doc["data"]["result"]:
        metric = series.get("metric", {})
        if match and not all(metric.get(k) == v for k, v in match.items()):
            continue
        ts  = [float(p[0]) for p in series["values"]]
        val = [float(p[1]) for p in series["values"]]
        return ts, val
    return [], []


def step6_plots(decisions):
    """Render the dissertation figures from the JSON/CSV already in OUTPUT_DIR."""
    log("Step 6: Rendering figures…")
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless — no display needed
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        log("  ⚠  matplotlib not installed — skipping figures (pip install matplotlib)")
        return

    def _to_dt(ts):
        return [datetime.datetime.fromtimestamp(t, datetime.timezone.utc).replace(tzinfo=None)
                for t in ts]

    # ── fig1: load-generator CPU demand curve ────────────────────────────────
    lg = _read_json("prometheus_load_generator_cpu.json")
    ts, cpu = _matrix_series(lg)
    if ts:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(_to_dt(ts), cpu, color="#1f77b4", lw=1.6, label="load-generator avg CPU")
        # Horizontal lines marking the CPU level that triggers each replica step.
        # desired = ceil(cpu*(1+0.15)/CPU_PER_REPLICA) -> boundary for N replicas at
        # cpu = (N-1)*CPU_PER_REPLICA/1.15 (approx; first boundary near 0).
        for n in (2, 3):
            level = (n - 1) * CPU_PER_REPLICA / 1.15
            ax.axhline(level, color="grey", ls="--", lw=0.8)
            ax.text(_to_dt(ts)[0], level, f" {n}-replica threshold",
                    va="bottom", ha="left", fontsize=8, color="grey")
        ax.set_title("Load-generator CPU demand (Alibaba ramp)")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("CPU (vCPU cores)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "fig1_cpu_demand_curve.png"), dpi=120)
        plt.close(fig)
        log("  ✓  Saved fig1_cpu_demand_curve.png")
    else:
        log("  ⚠  prometheus_load_generator_cpu.json empty — fig1 skipped")

    # ── fig2: replica count step plot ────────────────────────────────────────
    rep = _read_json("prometheus_replicas.json")
    ts, replicas = _matrix_series(rep, match={"deployment": "load-generator"})
    if ts:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.step(_to_dt(ts), replicas, where="post", color="#d62728", lw=1.8)
        ax.set_title("Autoscaler replica response (load-generator)")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Replicas")
        ax.set_ylim(0, max(replicas) + 1)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "fig2_replica_count.png"), dpi=120)
        plt.close(fig)
        log("  ✓  Saved fig2_replica_count.png")
    else:
        log("  ⚠  prometheus_replicas.json has no load-generator series — fig2 skipped")

    # ── fig3: current vs predicted CPU + replicas (predictive runs only) ──────
    if EVIDENCE_LABEL.lower() == "hpa" and any(d.get("method") == "predictive" for d in decisions):
        log("  ⚠  HPA-labelled run contains predictive decisions — fig3 skipped")
    elif decisions:
        idx       = list(range(len(decisions)))
        current   = [d.get("current_cpu")   for d in decisions]
        predicted = [d.get("predicted_cpu") for d in decisions]
        newrep    = [d.get("new_replicas")  for d in decisions]
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(idx, current,   color="#1f77b4", marker="o", ms=3, label="current CPU")
        ax1.plot(idx, predicted, color="#ff7f0e", marker="x", ms=4, ls="--",
                 label="predicted CPU (t+30m)")
        ax1.set_xlabel("Autoscaler decision #")
        ax1.set_ylabel("CPU (vCPU cores)")
        ax1.legend(loc="upper left", fontsize=8)
        ax2 = ax1.twinx()
        ax2.step(idx, newrep, where="post", color="#2ca02c", lw=1.2, alpha=0.6,
                 label="replicas")
        ax2.set_ylabel("Replicas", color="#2ca02c")
        ax2.set_ylim(0, (max(newrep) if newrep else 1) + 1)
        ax2.yaxis.get_major_locator().set_params(integer=True)
        ax1.set_title("Predictive autoscaler decisions: current vs predicted CPU")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "fig3_autoscaler_decisions.png"), dpi=120)
        plt.close(fig)
        log("  ✓  Saved fig3_autoscaler_decisions.png")
    else:
        log("  (no decisions — fig3 skipped, expected for the HPA pass)")


def _method_metrics(label):
    """Compute live SLA-violation % and over-provision % for one labelled pass."""
    base = os.path.join(BASE_EVIDENCE_DIR, label)
    rep  = _read_json("prometheus_replicas.json", base=base)
    cpu_doc = _read_json("prometheus_load_generator_cpu.json", base=base)
    r_ts, replicas = _matrix_series(rep, match={"deployment": "load-generator"})
    c_ts, cpu      = _matrix_series(cpu_doc)
    if not r_ts or not c_ts:
        return None

    # Align CPU samples onto replica timestamps (nearest replica value per CPU point).
    rep_by_ts = dict(zip(r_ts, replicas))
    r_sorted  = sorted(rep_by_ts)
    def replicas_at(t):
        # last replica value at or before t
        prev = r_sorted[0]
        for rt in r_sorted:
            if rt > t:
                break
            prev = rt
        return rep_by_ts[prev]

    viol = over = n = 0
    over_sum = 0.0
    for t, actual in zip(c_ts, cpu):
        provisioned = replicas_at(t) * CPU_PER_REPLICA
        n += 1
        if actual > provisioned:
            viol += 1
        if actual > 0:
            over_sum += (provisioned - actual) / actual
            over += 1
    return {
        "method":             label,
        "n_points":           n,
        "sla_violation_pct":  round(viol / n * 100, 2) if n else 0.0,
        "over_provision_pct": round(over_sum / over * 100, 2) if over else 0.0,
        "mean_cpu":           round(sum(cpu) / len(cpu), 4),
        "mean_replicas":      round(sum(replicas) / len(replicas), 2),
    }


def step7_compare():
    """If both labelled passes exist, emit predictive-vs-HPA comparison + figure."""
    pred_dir = os.path.join(BASE_EVIDENCE_DIR, "predictive")
    hpa_dir  = os.path.join(BASE_EVIDENCE_DIR, "hpa")
    if not (os.path.isdir(pred_dir) and os.path.isdir(hpa_dir)):
        return  # not a two-pass comparison run — nothing to do

    log("Step 7: Building predictive-vs-HPA comparison…")
    rows = [m for m in (_method_metrics("predictive"), _method_metrics("hpa")) if m]
    if len(rows) < 2:
        log("  ⚠  Missing Prometheus data in one pass — comparison skipped")
        return

    # comparison.csv lives at the evidence/ root (not inside a labelled folder)
    csv_path = os.path.join(BASE_EVIDENCE_DIR, "comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    log(f"  ✓  Saved comparison.csv  ({len(rows)} methods)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        log("  ⚠  matplotlib/numpy missing — comparison figure skipped")
        return

    methods = [r["method"] for r in rows]
    sla     = [r["sla_violation_pct"]  for r in rows]
    over    = [r["over_provision_pct"] for r in rows]
    x = np.arange(len(methods)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w/2, sla,  w, label="SLA violation %",  color="#d62728")
    ax.bar(x + w/2, over, w, label="Over-provision %", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("Percent")
    ax.set_title("Live cluster: predictive autoscaler vs Kubernetes HPA")
    ax.legend(fontsize=8)
    for i, (s, o) in enumerate(zip(sla, over)):
        ax.text(i - w/2, s, f"{s:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, o, f"{o:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_EVIDENCE_DIR, "fig4_predictive_vs_hpa.png"), dpi=120)
    plt.close(fig)
    log("  ✓  Saved fig4_predictive_vs_hpa.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global EVIDENCE_LABEL, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Collect GKE simulation evidence.")
    parser.add_argument("--label", default=os.getenv("EVIDENCE_LABEL", ""),
                        help="Sub-folder under evidence/ (e.g. predictive, hpa). "
                             "Also reads env EVIDENCE_LABEL.")
    args = parser.parse_args()
    EVIDENCE_LABEL = args.label.strip()
    OUTPUT_DIR = os.path.join(BASE_EVIDENCE_DIR, EVIDENCE_LABEL) if EVIDENCE_LABEL \
        else BASE_EVIDENCE_DIR

    setup()

    wait("letting the latest autoscaler CronJob run complete before collecting logs")
    raw_logs  = step1_collect_logs()

    wait("letting Prometheus metrics settle before querying")
    decisions = step2_extract_decisions(raw_logs)

    wait("waiting before Prometheus time-series export")
    step3_prometheus()

    wait("waiting before final cluster snapshot")
    step4_cluster_state()

    # No wait before summary/figures — just parsing already-collected data
    step5_summary(decisions)
    step6_plots(decisions)
    step7_compare()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{ts()} Interrupted by user — partial evidence saved to {OUTPUT_DIR}/")
        sys.exit(1)
