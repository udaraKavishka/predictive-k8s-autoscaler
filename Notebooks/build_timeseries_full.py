"""Build full Alibaba DLRM 5-minute time-series for research experiments.

This script converts event-based instance traces into a regular per-application
time-series that can be loaded by the research implementation notebooks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BuildConfig:
    data_path: Path
    output_path: Path
    sample_mode: bool
    sample_n_apps: int | None
    max_trace_seconds: float | None
    interval: int
    history_length: int
    forecast_steps: int


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(
        description="Build Alibaba CPU-demand time-series from event trace"
    )
    parser.add_argument(
        "--data-path",
        default="Alibaba/disaggregated_DLRM_trace.csv",
        help="Path to raw Alibaba trace CSV",
    )
    parser.add_argument(
        "--output-path",
        default="alibaba_timeseries_full.csv",
        help="Output path for built time-series CSV",
    )
    parser.add_argument(
        "--sample-mode",
        action="store_true",
        help="Enable sample mode (top-N apps and optional trace cap)",
    )
    parser.add_argument(
        "--sample-n-apps",
        type=int,
        default=10,
        help="Top N apps by instance count when sample mode is enabled",
    )
    parser.add_argument(
        "--max-trace-seconds",
        type=float,
        default=None,
        help="Optional trace cap in seconds (sample mode only)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Sampling interval in seconds (default: 300 = 5 min)",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=24,
        help="Look-back steps required by downstream model",
    )
    parser.add_argument(
        "--forecast-steps",
        type=int,
        default=6,
        help="Forecast horizon steps required by downstream model",
    )

    args = parser.parse_args()

    max_trace_seconds = args.max_trace_seconds
    if args.sample_mode and max_trace_seconds is None:
        max_trace_seconds = 86400.0

    return BuildConfig(
        data_path=Path(args.data_path),
        output_path=Path(args.output_path),
        sample_mode=args.sample_mode,
        sample_n_apps=args.sample_n_apps,
        max_trace_seconds=max_trace_seconds,
        interval=args.interval,
        history_length=args.history_length,
        forecast_steps=args.forecast_steps,
    )


def build_timeseries(cfg: BuildConfig) -> pd.DataFrame:
    print("=" * 72)
    print("Alibaba time-series builder")
    print(f"Mode              : {'SAMPLE' if cfg.sample_mode else 'FULL'}")
    print(f"Data path         : {cfg.data_path}")
    print(f"Output path       : {cfg.output_path}")
    print(f"Interval          : {cfg.interval}s")
    print(f"History length    : {cfg.history_length}")
    print(f"Forecast steps    : {cfg.forecast_steps}")
    if cfg.sample_mode:
        print(f"Sample apps       : top {cfg.sample_n_apps}")
        print(f"Max trace seconds : {cfg.max_trace_seconds}")
    print("=" * 72)

    df = pd.read_csv(cfg.data_path)
    print(f"Raw dataset shape : {df.shape}")
    print(f"Unique apps       : {df['app_name'].nunique()}")
    print(f"Role distribution :\n{df['role'].value_counts()}")

    df_work = df.copy()
    df_work["creation_time"] = df_work["creation_time"].fillna(0)
    full_trace_end = df_work["deletion_time"].dropna().max()
    df_work["deletion_time"] = df_work["deletion_time"].fillna(full_trace_end)

    if cfg.sample_mode and cfg.max_trace_seconds is not None:
        trace_end = float(cfg.max_trace_seconds)
        df_work = df_work[df_work["creation_time"] < trace_end].copy()
        df_work["deletion_time"] = df_work["deletion_time"].clip(upper=trace_end)
    else:
        trace_end = float(full_trace_end)

    df_work["role_encoded"] = (df_work["role"] == "HN").astype(int)
    df_work["max_instance_per_node"] = df_work["max_instance_per_node"].replace(
        -1, np.nan
    )

    if cfg.sample_mode and cfg.sample_n_apps:
        top_apps = (
            df_work.groupby("app_name")["instance_sn"]
            .count()
            .nlargest(cfg.sample_n_apps)
            .index
        )
        df_proc = df_work[df_work["app_name"].isin(top_apps)].copy()
        print(f"Selected apps      : {len(top_apps)}")
    else:
        df_proc = df_work.copy()
        print("Selected apps      : all")

    time_steps = np.arange(0, trace_end, cfg.interval)
    print(f"Trace end          : {trace_end:.0f}s ({trace_end / 3600:.1f}h)")
    print(f"Time bins          : {len(time_steps)}")
    print(f"Instances in scope : {len(df_proc):,}")

    ct = df_proc["creation_time"].to_numpy()
    dt = df_proc["deletion_time"].to_numpy()
    agg_records: list[pd.DataFrame] = []

    for idx, t in enumerate(time_steps):
        if idx % 50 == 0:
            print(f"  Bin {idx + 1}/{len(time_steps)} ({t / 3600:.1f}h)")
        active = df_proc[(ct <= t) & (dt > t)]
        if active.empty:
            continue

        agg = active.groupby("app_name", sort=False).agg(
            cpu_demand=("cpu_request", "sum"),
            gpu_request_mean=("gpu_request", "mean"),
            memory_request_mean=("memory_request", "mean"),
            rdma_request_mean=("rdma_request", "mean"),
            role_hn_fraction=("role_encoded", "mean"),
            instance_count=("cpu_request", "count"),
            max_instance_per_node=("max_instance_per_node", "mean"),
        )
        agg["timestamp"] = t
        agg_records.append(agg)

    if not agg_records:
        raise RuntimeError(
            "No aggregation records produced. Check input and configuration."
        )

    ts_df = pd.concat(agg_records).reset_index()
    ts_df = ts_df.sort_values(["app_name", "timestamp"]).reset_index(drop=True)
    ts_df["max_instance_per_node"] = ts_df["max_instance_per_node"].fillna(0)

    min_steps = cfg.history_length + cfg.forecast_steps + 10
    app_counts = ts_df.groupby("app_name")["timestamp"].count()
    valid_apps = app_counts[app_counts >= min_steps].index
    ts_df = ts_df[ts_df["app_name"].isin(valid_apps)].reset_index(drop=True)

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    ts_df.to_csv(cfg.output_path, index=False)

    print("\nBuild complete")
    print(f"Saved rows         : {len(ts_df):,}")
    print(f"Retained apps      : {ts_df['app_name'].nunique()}")
    print(f"Output file        : {cfg.output_path}")
    print(
        f"Time range         : {ts_df['timestamp'].min():.0f}s - {ts_df['timestamp'].max():.0f}s"
    )
    print("Suggested downstream setting:")
    print(f"  PREPROCESSED_PATH = '{cfg.output_path.as_posix()}'")

    return ts_df


def main() -> None:
    cfg = parse_args()
    build_timeseries(cfg)


if __name__ == "__main__":
    main()
