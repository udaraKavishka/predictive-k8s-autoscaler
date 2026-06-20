"""Research implementation V3 — targeted fixes for claim completion.

Changes from V2:
- EWC lambda: 500 → 100 (allows chunk-4 adaptation; fixes proposed vs Periodic Retrain)
- Replay ratio: 0.4 → 0.2 (reduces old-data dominance; fixes ablation F vs C/D)
- New stage: naive_ft — trains sequential model without EWC/ER for BWT comparison
- Multi-seed: adds paired t-test and Cohen's d vs each baseline → significance_tests.csv
- Baselines: loaded from cache by default (no --force-baselines needed)
"""

from __future__ import annotations

# Standard-library imports for filesystem operations, timing, and CLI parsing.
import argparse
import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Third-party imports for data handling, plotting, and ML/statistics.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import tensorflow as tf

# Keras components used for model architecture and custom continual-learning loss.
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    LSTM,
    concatenate,
)
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable

# Scikit-learn utilities used for scaling, grouping, and metrics.
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


# Dataclass that stores all experiment configuration in one typed object.
@dataclass
class ExperimentConfig:
    # Input/output locations.
    preprocessed_path: str = "alibaba_timeseries_full.csv"
    output_dir: str = "."

    # Core research parameters from notebook/proposal.
    interval: int = 300
    history_length: int = 24
    forecast_steps: int = 6
    n_chunks: int = 4
    epochs: int = 30
    batch_size: int = 32
    # V3: lowered from 500 → 100 to allow chunk-4 adaptation (fixes proposed > Periodic Retrain)
    ewc_lambda: float = 100.0
    replay_memory: int = 2000
    # V3: lowered from 0.4 → 0.2 so chunk-4 data dominates (fixes ablation F worst variant)
    replay_ratio: float = 0.2
    roll_window: int = 12
    sla_tol: float = 0.15

    # Randomness controls.
    seed: int = 42

    # Validation controls (option 1: full validation enabled by default).
    run_multi_seed: bool = True
    run_ablation: bool = True
    run_forgetting: bool = True
    run_naive_ft: bool = True
    run_cross_app: bool = True
    run_sensitivity: bool = True
    run_sla_analysis: bool = True
    run_dashboard: bool = True

    # Optional acceleration switches (kept off by default).
    fast_mode: bool = False

    # Resume/cache controls.
    resume: bool = True
    force_retrain: bool = False
    force_baselines: bool = False
    force_validation: bool = False
    refresh_all: bool = False

    # Execution control: run pipeline up to this named stage.
    run_until: str = "dashboard"


# Ordered stage names used for selective execution controls.
STAGE_ORDER = [
    "setup",
    "load",
    "features",
    "prepare",
    "train",
    "baselines",
    "core_outputs",
    "summary",
    "multi_seed",
    "ablation",
    "forgetting",
    "naive_ft",
    "cross_app",
    "sensitivity",
    "sla",
    "dashboard",
]


# Manifest filename that stores artifact metadata and config signature.
MANIFEST_FILE = "experiment_manifest.json"


# Lightweight progress logger to print clear terminal progress and timings.
class ProgressTracker:
    # Constructor stores expected total steps and initial timestamps.
    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self.current_step = 0
        self.script_start = time.perf_counter()

    # Starts a named step and returns a step-local start timestamp.
    def start(self, name: str) -> float:
        self.current_step += 1
        print(f"\n[{self.current_step:02d}/{self.total_steps:02d}] {name}")
        return time.perf_counter()

    # Ends a step and prints elapsed time in seconds.
    def end(self, step_start: float, note: str = "done") -> None:
        elapsed = time.perf_counter() - step_start
        print(f"    -> {note} ({elapsed:.2f}s)")

    # Prints final script runtime.
    def finish(self) -> None:
        total_elapsed = time.perf_counter() - self.script_start
        print(f"\nTotal execution time: {total_elapsed:.2f}s")


# Sets TensorFlow/NumPy randomness and CPU-only behavior for stable local execution.
def configure_environment(seed: int) -> None:
    # Reduce verbose TensorFlow C++ logs and force CPU-only mode.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Hide GPU devices in TensorFlow if visible.
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    # Configure deterministic random seeds.
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Keep CPU precision policy explicit and aligned to notebook CPU version.
    tf.keras.mixed_precision.set_global_policy("float32")


# Loads the preprocessed time-series CSV and validates required columns.
def load_preprocessed_timeseries(cfg: ExperimentConfig) -> pd.DataFrame:
    # Resolve and validate input file existence early.
    preprocessed_file = Path(cfg.preprocessed_path)
    if not preprocessed_file.exists():
        raise FileNotFoundError(
            f"Preprocessed file not found: {preprocessed_file}. "
            "Please keep alibaba_timeseries_full.csv in this folder or pass --preprocessed-path."
        )

    # Read CSV and verify core schema used by the training pipeline.
    ts_df = pd.read_csv(preprocessed_file)
    required_cols = {
        "app_name",
        "cpu_demand",
        "gpu_request_mean",
        "memory_request_mean",
        "rdma_request_mean",
        "role_hn_fraction",
        "max_instance_per_node",
        "timestamp",
    }
    missing = required_cols - set(ts_df.columns)
    if missing:
        raise ValueError(f"Preprocessed file missing columns: {sorted(missing)}")

    # Print concise dataset summary to the terminal.
    print(f"    loaded rows      : {len(ts_df):,}")
    print(f"    unique apps      : {ts_df['app_name'].nunique()}")
    print(
        f"    timestamp range  : {ts_df['timestamp'].min():.0f}s - {ts_df['timestamp'].max():.0f}s"
    )
    return ts_df


# Applies per-app temporal feature engineering (rolling/diff/cyclic features).
def engineer_features(ts_df: pd.DataFrame, roll_window: int) -> pd.DataFrame:
    # Internal helper that computes one app's derived features preserving time order.
    def _engineer_one_app(df_app: pd.DataFrame) -> pd.DataFrame:
        df_app = df_app.sort_values("timestamp").copy()
        df_app["cpu_diff"] = df_app["cpu_demand"].diff().fillna(0)
        df_app["cpu_roll_mean"] = (
            df_app["cpu_demand"].rolling(roll_window, min_periods=1).mean()
        )
        df_app["cpu_roll_std"] = (
            df_app["cpu_demand"].rolling(roll_window, min_periods=1).std().fillna(0)
        )
        df_app["cpu_roll_min"] = (
            df_app["cpu_demand"].rolling(roll_window, min_periods=1).min()
        )
        df_app["cpu_roll_max"] = (
            df_app["cpu_demand"].rolling(roll_window, min_periods=1).max()
        )
        seconds_in_day = 86400
        seconds_in_week = 604800
        df_app["hour_sin"] = np.sin(2 * np.pi * df_app["timestamp"] / seconds_in_day)
        df_app["hour_cos"] = np.cos(2 * np.pi * df_app["timestamp"] / seconds_in_day)
        df_app["dow_sin"] = np.sin(2 * np.pi * df_app["timestamp"] / seconds_in_week)
        df_app["dow_cos"] = np.cos(2 * np.pi * df_app["timestamp"] / seconds_in_week)
        return df_app

    # Apply feature engineering app-by-app to keep rolling stats app-specific.
    parts: list[pd.DataFrame] = []
    for app, grp in ts_df.groupby("app_name", sort=False):
        feat = _engineer_one_app(grp)
        feat["app_name"] = app
        parts.append(feat)

    # Concatenate all transformed apps back into one frame.
    out_df = pd.concat(parts, ignore_index=True)
    return out_df


# Converts a continuous app timeline into sequence samples for LSTM training.
def create_sequences(
    temporal_data: np.ndarray,
    static_data: np.ndarray,
    targets: np.ndarray,
    history_length: int,
    forecast_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Initialize output arrays as Python lists, then convert to NumPy for efficiency.
    X_seq, X_static, y = [], [], []
    total = len(temporal_data) - history_length - forecast_steps + 1
    for i in range(total):
        X_seq.append(temporal_data[i : i + history_length])
        X_static.append(static_data[i + history_length - 1])
        y.append(targets[i + history_length + forecast_steps - 1])
    return np.array(X_seq), np.array(X_static), np.array(y)


# Prepares full model inputs, scales them, saves scalers, and splits into chunks.
def prepare_model_inputs(
    ts_df: pd.DataFrame,
    cfg: ExperimentConfig,
    temporal_features: list[str],
    static_features: list[str],
    target: str,
    output_dir: Path,
) -> tuple[
    list[dict[str, np.ndarray]],
    StandardScaler,
    StandardScaler,
    StandardScaler,
    int,
    int,
]:
    # Ensure app ID feature exists for mixed-app sequence generation.
    app_labels = {app: i for i, app in enumerate(ts_df["app_name"].unique())}
    ts_df = ts_df.copy()
    ts_df["app_id"] = ts_df["app_name"].map(app_labels)
    static_features_full = static_features + ["app_id"]

    # Build sequences per app and combine into one training pool.
    all_x_seq, all_x_static, all_y = [], [], []
    min_steps_required = cfg.history_length + cfg.forecast_steps + 10
    for _, grp in ts_df.groupby("app_name", sort=False):
        grp = grp.sort_values("timestamp")
        if len(grp) < min_steps_required:
            continue
        t_arr = grp[temporal_features].values.astype(np.float32)
        s_arr = grp[static_features_full].values.astype(np.float32)
        y_arr = grp[target].values.astype(np.float32)
        x_seq, x_stat, y = create_sequences(
            t_arr, s_arr, y_arr, cfg.history_length, cfg.forecast_steps
        )
        all_x_seq.append(x_seq)
        all_x_static.append(x_stat)
        all_y.append(y)

    # Stop early with a clear message if no valid sequences were produced.
    if not all_x_seq:
        raise ValueError(
            "No sequences created. Check preprocessed data and sequence parameters."
        )

    # Concatenate data across apps for chunked continual training.
    x_seq_all = np.concatenate(all_x_seq, axis=0)
    x_static_all = np.concatenate(all_x_static, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    # Fit scalers on full data and transform arrays.
    n_temporal = x_seq_all.shape[2]
    n_static = x_static_all.shape[1]
    temporal_scaler = StandardScaler()
    static_scaler = StandardScaler()
    target_scaler = StandardScaler()
    x_seq_flat = x_seq_all.reshape(-1, n_temporal)
    temporal_scaler.fit(x_seq_flat)
    static_scaler.fit(x_static_all)
    target_scaler.fit(y_all.reshape(-1, 1))
    x_seq_scaled = temporal_scaler.transform(x_seq_flat).reshape(x_seq_all.shape)
    x_static_scaled = static_scaler.transform(x_static_all)
    y_scaled = target_scaler.transform(y_all.reshape(-1, 1)).flatten()

    # Persist scalers for future inference/reproducibility.
    with open(output_dir / "temporal_scaler.pkl", "wb") as f:
        pickle.dump(temporal_scaler, f)
    with open(output_dir / "static_scaler.pkl", "wb") as f:
        pickle.dump(static_scaler, f)
    with open(output_dir / "target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)

    # Split into sequential chunks (no shuffle) to simulate evolving workloads.
    chunk_size = len(y_scaled) // cfg.n_chunks
    chunks: list[dict[str, np.ndarray]] = []
    for i in range(cfg.n_chunks):
        s = i * chunk_size
        e = (i + 1) * chunk_size if i < cfg.n_chunks - 1 else len(y_scaled)
        chunks.append(
            {
                "X_seq": x_seq_scaled[s:e],
                "X_static": x_static_scaled[s:e],
                "y": y_scaled[s:e],
            }
        )

    # Print sequence preparation summary for terminal visibility.
    print(f"    total sequences : {len(y_scaled):,}")
    print(f"    sequence shape  : {x_seq_scaled.shape}")
    print(f"    static shape    : {x_static_scaled.shape}")
    print(f"    chunk sizes     : {[len(c['y']) for c in chunks]}")
    return chunks, temporal_scaler, static_scaler, target_scaler, n_temporal, n_static


# Builds the hybrid LSTM+MLP architecture used in the proposed method and baselines.
def create_hybrid_model(history_length: int, n_temporal: int, n_static: int) -> Model:
    # Define temporal branch input and stacked LSTM feature extractor.
    lstm_input = Input(shape=(history_length, n_temporal), name="temporal_input")
    x = LSTM(128, return_sequences=True, dropout=0.2, name="lstm_128")(lstm_input)
    x = LSTM(64, return_sequences=True, dropout=0.2, name="lstm_64")(x)
    x = LSTM(32, name="lstm_32")(x)
    lstm_embed = Dense(16, activation="relu", name="lstm_embedding")(x)

    # Define static branch input and dense representation pipeline.
    mlp_input = Input(shape=(n_static,), name="static_input")
    x = Dense(64, activation="relu", name="mlp_64")(mlp_input)
    x = BatchNormalization(name="mlp_bn")(x)
    x = Dropout(0.2, name="mlp_drop")(x)
    x = Dense(32, activation="relu", name="mlp_32")(x)
    mlp_embed = Dense(16, activation="relu", name="mlp_embedding")(x)

    # Fuse temporal and static embeddings and output one-step CPU forecast.
    fused = concatenate([lstm_embed, mlp_embed], name="fusion")
    fused = Dense(16, activation="relu", name="fusion_dense")(fused)
    output = Dense(1, activation="linear", name="cpu_forecast")(fused)

    # Compile with Adam + MSE/MAE as in notebook implementation.
    model = Model(inputs=[lstm_input, mlp_input], outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# Builds the LSTM-only baseline architecture.
def create_lstm_only_model(history_length: int, n_temporal: int) -> Model:
    # Define single-branch sequence model for baseline comparison.
    inp = Input(shape=(history_length, n_temporal))
    x = LSTM(128, return_sequences=True, dropout=0.2)(inp)
    x = LSTM(64, return_sequences=True, dropout=0.2)(x)
    x = LSTM(32)(x)
    out = Dense(1, activation="linear")(x)

    # Compile baseline with the same optimizer/loss family.
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# Custom helper that combines standard MSE with EWC regularization penalty.
@register_keras_serializable()
def loss_with_ewc(
    y_true: tf.Tensor, y_pred: tf.Tensor, model: Model, ewc_penalty_fn: Any
) -> tf.Tensor:
    # Compute base MSE and add current EWC penalty on trainable weights.
    mse = MeanSquaredError()(y_true, y_pred)
    penalty = ewc_penalty_fn(model.trainable_weights)
    return mse + penalty


# Elastic Weight Consolidation class for catastrophic forgetting mitigation.
class EWC:
    # Constructor stores model and initializes previous-knowledge placeholders.
    def __init__(self, model: Model, fisher_multiplier: float) -> None:
        self.model = model
        self.fisher_multiplier = fisher_multiplier
        self.fisher_diagonal: list[tf.Tensor] | None = None
        self.old_params: list[np.ndarray] | None = None

    # Estimates diagonal Fisher matrix using squared gradients on sampled examples.
    def compute_fisher_diagonal(
        self,
        x_lstm: np.ndarray,
        x_static: np.ndarray,
        y: np.ndarray,
        sample_size: int = 200,
    ) -> None:
        # Subsample data for faster Fisher approximation.
        n = len(y)
        if n > sample_size:
            idx = np.random.choice(n, sample_size, replace=False)
            x_lstm, x_static, y = x_lstm[idx], x_static[idx], y[idx]

        # Snapshot old weights and initialize zero Fisher accumulators.
        self.fisher_diagonal = [tf.zeros_like(w) for w in self.model.trainable_weights]
        self.old_params = [w.numpy().copy() for w in self.model.trainable_weights]

        # Compute gradients of average MSE and store squared values as Fisher diagonal.
        with tf.GradientTape() as tape:
            preds = self.model([x_lstm, x_static], training=False)
            loss = tf.reduce_mean(
                tf.square(tf.cast(y.reshape(-1, 1), tf.float32) - preds)
            )
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.fisher_diagonal = [
            tf.square(g) if g is not None else tf.zeros_like(w)
            for g, w in zip(grads, self.model.trainable_weights)
        ]

    # Computes scalar EWC penalty from current weights vs old protected weights.
    def ewc_loss(self, current_weights: list[tf.Tensor]) -> tf.Tensor:
        # Return zero if no previous-task Fisher exists (first chunk).
        if self.fisher_diagonal is None or self.old_params is None:
            return tf.constant(0.0, dtype=tf.float32)

        # Sum weighted quadratic distance over all trainable parameters.
        penalty = tf.constant(0.0, dtype=tf.float32)
        for f_i, old_w, cur_w in zip(
            self.fisher_diagonal, self.old_params, current_weights
        ):
            penalty += tf.reduce_sum(f_i * tf.square(cur_w - old_w))
        return 0.5 * self.fisher_multiplier * penalty

    # Fits the model using the MSE + EWC composite loss.
    def train_with_ewc(
        self,
        x_lstm: np.ndarray,
        x_static: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        validation_data: tuple[list[np.ndarray], np.ndarray] | None = None,
    ) -> Any:
        # Build closure loss so Keras can call penalty during optimization.
        def composite_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return loss_with_ewc(y_true, y_pred, self.model, self.ewc_loss)

        # Re-compile model with composite loss and train.
        self.model.compile(optimizer="adam", loss=composite_loss, metrics=["mae"])
        history = self.model.fit(
            [x_lstm, x_static],
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1,
        )

        # Restore plain MSE compile for regular prediction/evaluation paths.
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return history


# Experience Replay memory for mixing historical samples into new chunk training.
class ExperienceReplay:
    # Constructor initializes bounded memory buffers and replay ratio.
    def __init__(self, memory_size: int, replay_ratio: float) -> None:
        self.memory_size = memory_size
        self.replay_ratio = replay_ratio
        self._mem_lstm: list[np.ndarray] = []
        self._mem_static: list[np.ndarray] = []
        self._mem_y: list[float] = []

    # Convenience property exposing current replay memory size.
    @property
    def size(self) -> int:
        return len(self._mem_y)

    # Updates replay memory with current chunk training samples.
    def update_memory(
        self, x_lstm: np.ndarray, x_static: np.ndarray, y: np.ndarray
    ) -> None:
        # Append samples and truncate to fixed capacity via random keep set.
        self._mem_lstm.extend([row for row in x_lstm])
        self._mem_static.extend([row for row in x_static])
        self._mem_y.extend([float(v) for v in y])
        if self.size > self.memory_size:
            keep = np.random.choice(self.size, self.memory_size, replace=False)
            self._mem_lstm = [self._mem_lstm[i] for i in keep]
            self._mem_static = [self._mem_static[i] for i in keep]
            self._mem_y = [self._mem_y[i] for i in keep]

    # Returns a replay batch proportional to current batch size.
    def get_replay_batch(
        self, n_new_samples: int
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        # Compute replay sample count and handle empty-memory case.
        n_replay = min(int(n_new_samples * self.replay_ratio), self.size)
        if n_replay == 0:
            return None, None, None

        # Randomly sample replay memory and return NumPy arrays.
        idx = np.random.choice(self.size, n_replay, replace=False)
        replay_lstm = np.array([self._mem_lstm[i] for i in idx])
        replay_static = np.array([self._mem_static[i] for i in idx])
        replay_y = np.array([self._mem_y[i] for i in idx])
        return replay_lstm, replay_static, replay_y

    # Mixes current chunk data with replay data and returns combined arrays.
    def mix_with_replay(
        self, x_lstm: np.ndarray, x_static: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Sample replay and pass-through current batch if replay unavailable.
        r_lstm, r_stat, r_y = self.get_replay_batch(len(y))
        if r_lstm is None:
            return x_lstm, x_static, y

        # Concatenate current and replay arrays for joint training.
        mixed_lstm = np.concatenate([x_lstm, r_lstm])
        mixed_static = np.concatenate([x_static, r_stat])
        mixed_y = np.concatenate([y.flatten(), r_y.flatten()])
        return mixed_lstm, mixed_static, mixed_y


# Computes cost/risk metrics from predictions and true values.
# OverProv and UnderProv are expressed as a percentage of total CPU demand so
# they are comparable across methods regardless of dataset size or raw unit scale.
# SLA violation applies the 15% tolerance band from the EVALUATION_GUIDE.
def compute_provisioning_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, sla_tol: float = 0.15
) -> tuple[float, float, float]:
    diff = y_pred - y_true
    total_demand = float(np.sum(np.abs(y_true))) + 1e-8
    over_provision = float(np.sum(np.maximum(diff, 0)) / total_demand * 100)
    under_provision = float(np.sum(np.maximum(-diff, 0)) / total_demand * 100)
    # Under-provision only counts when prediction is below the SLA tolerance band.
    sla_violation_pct = float(np.mean(y_pred < (1.0 - sla_tol) * y_true) * 100)
    return over_provision, under_provision, sla_violation_pct


# Main continual-learning training loop for proposed EWC+ER model.
def train_proposed_model(
    cfg: ExperimentConfig,
    chunks: list[dict[str, np.ndarray]],
    n_temporal: int,
    n_static: int,
    target_scaler: StandardScaler,
) -> tuple[
    Model,
    list[dict[str, float]],
    dict[int, dict[int, float]],
    list[Any],
    list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    np.ndarray,
    np.ndarray,
]:
    # Initialize model, continual-learning helpers, and result collectors.
    model = create_hybrid_model(cfg.history_length, n_temporal, n_static)
    ewc = EWC(model, fisher_multiplier=cfg.ewc_lambda)
    replay = ExperienceReplay(
        memory_size=cfg.replay_memory, replay_ratio=cfg.replay_ratio
    )
    chunk_results: list[dict[str, float]] = []
    bwt_matrix: dict[int, dict[int, float]] = {}
    train_histories: list[Any] = []
    val_sets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    prev_x_seq, prev_x_static, prev_y = None, None, None

    # Iterate through chronological chunks and perform continual-learning steps.
    for chunk_idx, chunk in enumerate(chunks):
        chunk_start = time.perf_counter()
        print(f"    [chunk {chunk_idx + 1}/{cfg.n_chunks}] preparing split")
        x_seq = chunk["X_seq"]
        x_static = chunk["X_static"]
        y = chunk["y"]

        # Create 80/20 time-ordered split to avoid leakage across future samples.
        split = int(0.8 * len(y))
        x_seq_tr, x_seq_val = x_seq[:split], x_seq[split:]
        x_stat_tr, x_stat_val = x_static[:split], x_static[split:]
        y_tr, y_val = y[:split], y[split:]
        val_sets.append((x_seq_val, x_stat_val, y_val))
        print(f"      train={len(y_tr):,} | val={len(y_val):,}")

        # Compute Fisher from previous chunk before current EWC training (chunk > 1).
        if (
            chunk_idx > 0
            and prev_x_seq is not None
            and prev_x_static is not None
            and prev_y is not None
        ):
            print("      computing Fisher information (EWC)")
            ewc.compute_fisher_diagonal(
                prev_x_seq, prev_x_static, prev_y, sample_size=200
            )

        # Mix replay memory samples with current chunk train data.
        x_seq_mix, x_stat_mix, y_mix = replay.mix_with_replay(x_seq_tr, x_stat_tr, y_tr)
        print(f"      training samples after replay mix: {len(y_mix):,}")

        # Train first chunk with plain MSE; later chunks with EWC composite loss.
        if chunk_idx == 0:
            print("      training mode: plain MSE (bootstrap chunk)")
            history = model.fit(
                [x_seq_mix, x_stat_mix],
                y_mix,
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                validation_data=([x_seq_val, x_stat_val], y_val),
                verbose=1,
            )
        else:
            print(f"      training mode: EWC + ER (lambda={cfg.ewc_lambda})")
            history = ewc.train_with_ewc(
                x_seq_mix,
                x_stat_mix,
                y_mix,
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                validation_data=([x_seq_val, x_stat_val], y_val),
            )
        train_histories.append(history)

        # Evaluate current chunk on real-value scale to report business-meaningful metrics.
        y_pred_scaled = model.predict([x_seq_val, x_stat_val], verbose=0).flatten()
        y_pred_real = target_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).flatten()
        y_val_real = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        mae = float(mean_absolute_error(y_val_real, y_pred_real))
        rmse = float(np.sqrt(mean_squared_error(y_val_real, y_pred_real)))
        mape = float(
            np.mean(np.abs((y_val_real - y_pred_real) / (y_val_real + 1e-8))) * 100
        )
        chunk_results.append(
            {"chunk": chunk_idx + 1, "mae": mae, "rmse": rmse, "mape": mape}
        )
        print(f"      metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")

        # Re-evaluate earlier chunks for backward transfer (forgetting) tracking.
        bwt_matrix[chunk_idx] = {}
        for prev_idx, (px_seq, px_stat, py) in enumerate(val_sets[:-1]):
            py_pred_scaled = model.predict([px_seq, px_stat], verbose=0).flatten()
            py_pred_real = target_scaler.inverse_transform(
                py_pred_scaled.reshape(-1, 1)
            ).flatten()
            py_real = target_scaler.inverse_transform(py.reshape(-1, 1)).flatten()
            bwt_mae = float(mean_absolute_error(py_real, py_pred_real))
            bwt_matrix[chunk_idx][prev_idx] = bwt_mae
            print(f"      BWT eval on chunk {prev_idx + 1}: MAE={bwt_mae:.4f}")

        # Update replay memory and cache current training set for next Fisher step.
        replay.update_memory(x_seq_tr, x_stat_tr, y_tr)
        prev_x_seq, prev_x_static, prev_y = (
            x_seq_tr.copy(),
            x_stat_tr.copy(),
            y_tr.copy(),
        )
        print(f"      replay buffer size: {replay.size}/{cfg.replay_memory}")
        print(f"      chunk elapsed: {time.perf_counter() - chunk_start:.2f}s")

    # Evaluate final chunk predictions once for downstream baseline and SLA analysis.
    x_final_seq, x_final_stat, y_final = val_sets[-1]
    y_final_pred_scaled = model.predict(
        [x_final_seq, x_final_stat], verbose=0
    ).flatten()
    y_final_pred = target_scaler.inverse_transform(
        y_final_pred_scaled.reshape(-1, 1)
    ).flatten()
    y_final_real = target_scaler.inverse_transform(y_final.reshape(-1, 1)).flatten()
    return (
        model,
        chunk_results,
        bwt_matrix,
        train_histories,
        val_sets,
        y_final_real,
        y_final_pred,
    )


# Runs all baselines and returns predictions and summary table on final chunk.
def run_baselines(
    cfg: ExperimentConfig,
    chunks: list[dict[str, np.ndarray]],
    n_temporal: int,
    n_static: int,
    x_final_seq: np.ndarray,
    x_final_stat: np.ndarray,
    y_final_real: np.ndarray,
    y_final_pred: np.ndarray,
    target_scaler: StandardScaler,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Model]]:
    # Train chunk-1 based static baselines and periodic-retrain baseline.
    x_c1_seq, x_c1_stat, y_c1 = (
        chunks[0]["X_seq"],
        chunks[0]["X_static"],
        chunks[0]["y"],
    )
    split = int(0.8 * len(y_c1))
    x_tr_seq, x_tr_stat, y_tr_c1 = x_c1_seq[:split], x_c1_stat[:split], y_c1[:split]

    # Baseline 1: HPA-reactive approximation using the same forecast horizon as neural
    # models (forecast_steps steps ahead) so the comparison is on equal footing.
    # A 1-step lag gives artificially near-zero MAE because consecutive readings change
    # very little; shifting by forecast_steps reflects the real scheduling delay.
    y_hpa_pred = np.roll(y_final_real, cfg.forecast_steps)
    y_hpa_pred[: cfg.forecast_steps] = y_final_real[: cfg.forecast_steps]

    # Baseline 2: Static LSTM trained once on chunk 1.
    static_lstm_model = create_lstm_only_model(cfg.history_length, n_temporal)
    static_lstm_model.fit(
        x_tr_seq, y_tr_c1, epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=0
    )
    y_slstm_scaled = static_lstm_model.predict(x_final_seq, verbose=0).flatten()
    y_slstm = target_scaler.inverse_transform(y_slstm_scaled.reshape(-1, 1)).flatten()

    # Baseline 3: Static hybrid trained once on chunk 1.
    static_hybrid_model = create_hybrid_model(cfg.history_length, n_temporal, n_static)
    static_hybrid_model.fit(
        [x_tr_seq, x_tr_stat],
        y_tr_c1,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=0,
    )
    y_shybrid_scaled = static_hybrid_model.predict(
        [x_final_seq, x_final_stat], verbose=0
    ).flatten()
    y_shybrid = target_scaler.inverse_transform(
        y_shybrid_scaled.reshape(-1, 1)
    ).flatten()

    # Baseline 4: Periodic retrain from scratch per chunk (sequential retraining).
    periodic_model = create_hybrid_model(cfg.history_length, n_temporal, n_static)
    for idx, chunk in enumerate(chunks):
        print(f"    periodic retrain chunk {idx + 1}/{len(chunks)}")
        sp = int(0.8 * len(chunk["y"]))
        periodic_model.fit(
            [chunk["X_seq"][:sp], chunk["X_static"][:sp]],
            chunk["y"][:sp],
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            verbose=0,
        )
    y_periodic_scaled = periodic_model.predict(
        [x_final_seq, x_final_stat], verbose=0
    ).flatten()
    y_periodic = target_scaler.inverse_transform(
        y_periodic_scaled.reshape(-1, 1)
    ).flatten()

    # Build helper for one-row metric computation across methods.
    def summary_row(
        name: str, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float | str]:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        # Raw MAPE is inflated by near-zero cpu_demand values; report filtered MAPE
        # on samples above the 10th percentile of non-zero demand as primary %.
        nonzero_mask = y_true > 0
        p10 = float(np.percentile(y_true[nonzero_mask], 10)) if nonzero_mask.any() else 0.0
        active_mask = y_true > p10
        if active_mask.any():
            mape = float(
                np.mean(np.abs((y_true[active_mask] - y_pred[active_mask])
                               / (y_true[active_mask] + 1e-8))) * 100
            )
        else:
            mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
        op, up, sla = compute_provisioning_metrics(y_true, y_pred)
        return {
            "Method": name,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE% (filtered)": mape,
            "OverProv%": op,
            "UnderProv%": up,
            "SLA_Viol%": sla,
        }

    # Assemble baseline comparison table including proposed model.
    rows = [
        summary_row("HPA-Reactive", y_final_real, y_hpa_pred),
        summary_row("Static LSTM", y_final_real, y_slstm),
        summary_row("Static Hybrid", y_final_real, y_shybrid),
        summary_row("Periodic Retrain", y_final_real, y_periodic),
        summary_row("Proposed (EWC+ER)", y_final_real, y_final_pred),
    ]
    results_df = pd.DataFrame(rows).set_index("Method")

    # Return both tables and predictions for downstream visualizations/validations.
    preds = {
        "HPA-Reactive": y_hpa_pred,
        "Static LSTM": y_slstm,
        "Static Hybrid": y_shybrid,
        "Periodic Retrain": y_periodic,
        "Proposed (EWC+ER)": y_final_pred,
    }
    models = {
        "static_lstm_model": static_lstm_model,
        "static_hybrid_model": static_hybrid_model,
        "periodic_model": periodic_model,
    }
    return results_df, preds, models


# Creates validation splits from prepared chunks without running training.
def build_val_sets_from_chunks(
    chunks: list[dict[str, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    # Recreate deterministic 80/20 validation splits used during training.
    val_sets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for chunk in chunks:
        split = int(0.8 * len(chunk["y"]))
        x_seq_val = chunk["X_seq"][split:]
        x_stat_val = chunk["X_static"][split:]
        y_val = chunk["y"][split:]
        val_sets.append((x_seq_val, x_stat_val, y_val))
    return val_sets


# Runs or reuses each baseline independently with artifact-aware status logging.
def run_baselines_with_cache(
    cfg: ExperimentConfig,
    chunks: list[dict[str, np.ndarray]],
    n_temporal: int,
    n_static: int,
    x_final_seq: np.ndarray,
    x_final_stat: np.ndarray,
    y_final_real: np.ndarray,
    y_final_pred: np.ndarray,
    target_scaler: StandardScaler,
    output_dir: Path,
    manifest: dict[str, Any],
    signature_matches: bool,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Model], dict[str, str]]:
    # Track baseline stage status per model for detailed terminal reporting.
    baseline_status: dict[str, str] = {}
    preds: dict[str, np.ndarray] = {}
    models: dict[str, Model] = {}

    # Always compute HPA prediction quickly as reactive baseline reference.
    # Use forecast_steps shift so the comparison horizon matches neural models.
    print("    [baseline][hpa] computing reactive predictor")
    y_hpa_pred = np.roll(y_final_real, cfg.forecast_steps)
    y_hpa_pred[: cfg.forecast_steps] = y_final_real[: cfg.forecast_steps]
    preds["HPA-Reactive"] = y_hpa_pred
    baseline_status["HPA-Reactive"] = "computed"

    # Shared chunk-1 split used by static baselines.
    x_c1_seq, x_c1_stat, y_c1 = (
        chunks[0]["X_seq"],
        chunks[0]["X_static"],
        chunks[0]["y"],
    )
    split = int(0.8 * len(y_c1))
    x_tr_seq, x_tr_stat, y_tr_c1 = x_c1_seq[:split], x_c1_stat[:split], y_c1[:split]

    # Baseline 1: Static LSTM cache paths.
    slstm_model_path = output_dir / "baseline_static_lstm.keras"
    slstm_pred_path = output_dir / "baseline_static_lstm_pred.npz"
    if (
        cfg.resume
        and signature_matches
        and (not cfg.force_baselines and not cfg.refresh_all)
        and artifacts_exist([slstm_model_path, slstm_pred_path])
    ):
        print("    [baseline][static_lstm] found saved model+prediction -> loading")
        models["static_lstm_model"] = tf.keras.models.load_model(slstm_model_path)
        slstm_npz = np.load(slstm_pred_path)
        preds["Static LSTM"] = slstm_npz["y_pred"]
        baseline_status["Static LSTM"] = "loaded"
    else:
        print("    [baseline][static_lstm] training started")
        static_lstm_model = create_lstm_only_model(cfg.history_length, n_temporal)
        static_lstm_model.fit(
            x_tr_seq, y_tr_c1, epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=0
        )
        y_slstm_scaled = static_lstm_model.predict(x_final_seq, verbose=0).flatten()
        y_slstm = target_scaler.inverse_transform(
            y_slstm_scaled.reshape(-1, 1)
        ).flatten()
        static_lstm_model.save(slstm_model_path)
        np.savez(slstm_pred_path, y_pred=y_slstm)
        models["static_lstm_model"] = static_lstm_model
        preds["Static LSTM"] = y_slstm
        baseline_status["Static LSTM"] = "trained"
    mark_artifact(manifest, "baseline_static_lstm_model", slstm_model_path)
    mark_artifact(manifest, "baseline_static_lstm_pred", slstm_pred_path)

    # Baseline 2: Static Hybrid cache paths.
    shybrid_model_path = output_dir / "baseline_static_hybrid.keras"
    shybrid_pred_path = output_dir / "baseline_static_hybrid_pred.npz"
    if (
        cfg.resume
        and signature_matches
        and (not cfg.force_baselines and not cfg.refresh_all)
        and artifacts_exist([shybrid_model_path, shybrid_pred_path])
    ):
        print("    [baseline][static_hybrid] found saved model+prediction -> loading")
        models["static_hybrid_model"] = tf.keras.models.load_model(shybrid_model_path)
        shybrid_npz = np.load(shybrid_pred_path)
        preds["Static Hybrid"] = shybrid_npz["y_pred"]
        baseline_status["Static Hybrid"] = "loaded"
    else:
        print("    [baseline][static_hybrid] training started")
        static_hybrid_model = create_hybrid_model(
            cfg.history_length, n_temporal, n_static
        )
        static_hybrid_model.fit(
            [x_tr_seq, x_tr_stat],
            y_tr_c1,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            verbose=0,
        )
        y_shybrid_scaled = static_hybrid_model.predict(
            [x_final_seq, x_final_stat], verbose=0
        ).flatten()
        y_shybrid = target_scaler.inverse_transform(
            y_shybrid_scaled.reshape(-1, 1)
        ).flatten()
        static_hybrid_model.save(shybrid_model_path)
        np.savez(shybrid_pred_path, y_pred=y_shybrid)
        models["static_hybrid_model"] = static_hybrid_model
        preds["Static Hybrid"] = y_shybrid
        baseline_status["Static Hybrid"] = "trained"
    mark_artifact(manifest, "baseline_static_hybrid_model", shybrid_model_path)
    mark_artifact(manifest, "baseline_static_hybrid_pred", shybrid_pred_path)

    # Baseline 3: Periodic retrain cache paths.
    periodic_model_path = output_dir / "baseline_periodic.keras"
    periodic_pred_path = output_dir / "baseline_periodic_pred.npz"
    if (
        cfg.resume
        and signature_matches
        and (not cfg.force_baselines and not cfg.refresh_all)
        and artifacts_exist([periodic_model_path, periodic_pred_path])
    ):
        print("    [baseline][periodic] found saved model+prediction -> loading")
        models["periodic_model"] = tf.keras.models.load_model(periodic_model_path)
        periodic_npz = np.load(periodic_pred_path)
        preds["Periodic Retrain"] = periodic_npz["y_pred"]
        baseline_status["Periodic Retrain"] = "loaded"
    else:
        print("    [baseline][periodic] training started")
        periodic_model = create_hybrid_model(cfg.history_length, n_temporal, n_static)
        for idx, chunk in enumerate(chunks):
            print(f"      [baseline][periodic] sub-step chunk {idx + 1}/{len(chunks)}")
            sp = int(0.8 * len(chunk["y"]))
            periodic_model.fit(
                [chunk["X_seq"][:sp], chunk["X_static"][:sp]],
                chunk["y"][:sp],
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                verbose=0,
            )
        y_periodic_scaled = periodic_model.predict(
            [x_final_seq, x_final_stat], verbose=0
        ).flatten()
        y_periodic = target_scaler.inverse_transform(
            y_periodic_scaled.reshape(-1, 1)
        ).flatten()
        periodic_model.save(periodic_model_path)
        np.savez(periodic_pred_path, y_pred=y_periodic)
        models["periodic_model"] = periodic_model
        preds["Periodic Retrain"] = y_periodic
        baseline_status["Periodic Retrain"] = "trained"
    mark_artifact(manifest, "baseline_periodic_model", periodic_model_path)
    mark_artifact(manifest, "baseline_periodic_pred", periodic_pred_path)

    # Always include proposed prediction passed from proposed-training stage.
    preds["Proposed (EWC+ER)"] = y_final_pred

    # Create one-row metrics table helper used for all methods.
    def summary_row(
        name: str, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float | str]:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        # Raw MAPE is inflated by near-zero cpu_demand values; report filtered MAPE
        # on samples above the 10th percentile of non-zero demand as primary %.
        nonzero_mask = y_true > 0
        p10 = float(np.percentile(y_true[nonzero_mask], 10)) if nonzero_mask.any() else 0.0
        active_mask = y_true > p10
        if active_mask.any():
            mape = float(
                np.mean(np.abs((y_true[active_mask] - y_pred[active_mask])
                               / (y_true[active_mask] + 1e-8))) * 100
            )
        else:
            mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
        op, up, sla = compute_provisioning_metrics(y_true, y_pred)
        return {
            "Method": name,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE% (filtered)": mape,
            "OverProv%": op,
            "UnderProv%": up,
            "SLA_Viol%": sla,
        }

    # Build final baseline comparison table.
    rows = [
        summary_row("HPA-Reactive", y_final_real, preds["HPA-Reactive"]),
        summary_row("Static LSTM", y_final_real, preds["Static LSTM"]),
        summary_row("Static Hybrid", y_final_real, preds["Static Hybrid"]),
        summary_row("Periodic Retrain", y_final_real, preds["Periodic Retrain"]),
        summary_row("Proposed (EWC+ER)", y_final_real, preds["Proposed (EWC+ER)"]),
    ]
    results_df = pd.DataFrame(rows).set_index("Method")

    return results_df, preds, models, baseline_status


# Saves chunk and baseline tables to disk for reproducibility.
def save_tables(
    output_dir: Path, chunk_results: list[dict[str, float]], results_df: pd.DataFrame
) -> None:
    # Write per-chunk metrics and baseline summary in CSV and JSON formats.
    chunk_df = pd.DataFrame(chunk_results)
    chunk_df.to_csv(output_dir / "chunk_metrics.csv", index=False)
    results_df.to_csv(output_dir / "baseline_results.csv")
    with open(output_dir / "chunk_metrics.json", "w", encoding="utf-8") as f:
        json.dump(chunk_results, f, indent=2)
    with open(output_dir / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(results_df.reset_index().to_dict(orient="records"), f, indent=2)


# Produces the main six-panel evaluation figure.
def plot_evaluation_results(
    output_dir: Path,
    chunk_results: list[dict[str, float]],
    bwt_matrix: dict[int, dict[int, float]],
    y_final_real: np.ndarray,
    preds: dict[str, np.ndarray],
    results_df: pd.DataFrame,
    train_histories: list[Any],
) -> None:
    # Create figure and allocate six axes for the core evaluation dashboard.
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Adaptive CPU Auto-Scaling: Hybrid LSTM+MLP with EWC + ER\nAlibaba DLRM Trace — Core Evaluation",
        fontsize=13,
        fontweight="bold",
    )

    # Plot 1: per-chunk MAE and RMSE progression.
    ax = axes[0, 0]
    labels = [f"Chunk {r['chunk']}" for r in chunk_results]
    ax.plot(
        labels, [r["mae"] for r in chunk_results], marker="o", label="MAE", linewidth=2
    )
    ax.plot(
        labels,
        [r["rmse"] for r in chunk_results],
        marker="s",
        label="RMSE",
        linewidth=2,
    )
    ax.set_title("Prediction Error Across Chunks")
    ax.set_ylabel("Error (CPU cores)")
    ax.grid(alpha=0.4)
    ax.legend()

    # Plot 2: backward transfer trajectories.
    ax = axes[0, 1]
    for after_chunk in sorted(bwt_matrix.keys()):
        vals = [
            bwt_matrix[after_chunk][j] for j in sorted(bwt_matrix[after_chunk].keys())
        ]
        labs = [f"C{j + 1}" for j in sorted(bwt_matrix[after_chunk].keys())]
        if vals:
            ax.plot(
                labs, vals, marker="D", linewidth=2, label=f"After C{after_chunk + 1}"
            )
    ax.set_title("Backward Transfer (Forgetting)")
    ax.set_ylabel("MAE on Earlier Chunks")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=8)

    # Plot 3: predicted vs actual on final chunk sample.
    ax = axes[0, 2]
    n_show = min(200, len(y_final_real))
    ax.plot(y_final_real[:n_show], color="black", linewidth=1.3, label="Actual")
    ax.plot(
        preds["Proposed (EWC+ER)"][:n_show],
        color="steelblue",
        linewidth=1,
        label="Proposed",
    )
    ax.plot(
        preds["Periodic Retrain"][:n_show],
        color="tomato",
        linestyle="--",
        linewidth=1,
        label="Periodic",
    )
    ax.plot(
        preds["Static Hybrid"][:n_show],
        color="purple",
        linestyle=":",
        linewidth=1,
        label="Static Hybrid",
    )
    ax.set_title("Predicted vs Actual (Final Chunk)")
    ax.set_ylabel("CPU Demand")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=8)

    # Plot 4: baseline MAE bar chart.
    ax = axes[1, 0]
    methods = results_df.index.tolist()
    mae_vals = results_df["MAE"].values
    colors = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60", "#2980b9"]
    ax.bar(methods, mae_vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_title("Final Chunk MAE: Baselines")
    ax.set_ylabel("MAE")
    ax.tick_params(axis="x", rotation=15, labelsize=8)
    ax.grid(axis="y", alpha=0.4)

    # Plot 5: baseline SLA violation bar chart.
    ax = axes[1, 1]
    ax.bar(
        methods,
        results_df["SLA_Viol%"].values,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_title("SLA Violation Rate")
    ax.set_ylabel("SLA Viol. (%)")
    ax.tick_params(axis="x", rotation=15, labelsize=8)
    ax.grid(axis="y", alpha=0.4)

    # Plot 6: training and validation loss per chunk.
    ax = axes[1, 2]
    chunk_colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    for i, hist in enumerate(train_histories):
        c = chunk_colors[i % len(chunk_colors)]
        ax.plot(
            hist.history.get("loss", []),
            color=c,
            linewidth=1.5,
            label=f"C{i + 1} train",
        )
        if "val_loss" in hist.history:
            ax.plot(
                hist.history["val_loss"],
                color=c,
                linestyle="--",
                linewidth=1.3,
                label=f"C{i + 1} val",
            )
    ax.set_title("Training Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=7, ncol=2)

    # Save and close figure to release memory.
    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# Computes and returns scalar backward transfer from matrix and per-chunk results.
def compute_bwt_scalar(
    bwt_mat: dict[int, dict[int, float]], chunk_results: list[dict[str, float]]
) -> float:
    # Use official formula terms R(T,j)-R(j,j) over prior chunks.
    scores = []
    n_chunks = len(chunk_results)
    for j in range(n_chunks - 1):
        r_jj = chunk_results[j]["mae"]
        r_last = bwt_mat.get(n_chunks - 1, {}).get(j)
        if r_last is not None:
            scores.append(r_last - r_jj)
    return float(np.mean(scores)) if scores else float("nan")


# Trains a naive sequential fine-tuning model (no EWC, no ER) for BWT comparison.
# Used to demonstrate that EWC+ER retains earlier-chunk knowledge better than plain FT.
def run_naive_ft_comparison(
    cfg: ExperimentConfig,
    chunks: list[dict[str, np.ndarray]],
    n_temporal: int,
    n_static: int,
    target_scaler: StandardScaler,
) -> dict[str, Any]:
    import copy
    vcfg = copy.copy(cfg)
    vcfg.ewc_lambda = 0.0
    vcfg.replay_ratio = 0.0
    vcfg.replay_memory = 0

    model = create_hybrid_model(cfg.history_length, n_temporal, n_static)
    diagonal_maes: list[float] = []
    val_sets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for chunk_idx, chunk in enumerate(chunks):
        split = int(0.8 * len(chunk["y"]))
        x_seq_tr, x_seq_val = chunk["X_seq"][:split], chunk["X_seq"][split:]
        x_stat_tr, x_stat_val = chunk["X_static"][:split], chunk["X_static"][split:]
        y_tr, y_val = chunk["y"][:split], chunk["y"][split:]
        val_sets.append((x_seq_val, x_stat_val, y_val))

        print(f"    [naive_ft] chunk {chunk_idx + 1}/{cfg.n_chunks} — plain MSE fine-tuning")
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        model.fit(
            [x_seq_tr, x_stat_tr],
            y_tr,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            validation_data=([x_seq_val, x_stat_val], y_val),
            verbose=0,
        )

        y_pred_s = model.predict([x_seq_val, x_stat_val], verbose=0).flatten()
        y_pred_r = target_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
        y_val_r = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        diag_mae = float(mean_absolute_error(y_val_r, y_pred_r))
        diagonal_maes.append(diag_mae)
        print(f"      diagonal MAE (chunk {chunk_idx + 1}): {diag_mae:.4f}")

    # Re-evaluate final model on all earlier chunk validation sets.
    final_maes: list[float] = []
    for prev_idx, (px_seq, px_stat, py) in enumerate(val_sets[:-1]):
        py_pred_s = model.predict([px_seq, px_stat], verbose=0).flatten()
        py_pred_r = target_scaler.inverse_transform(py_pred_s.reshape(-1, 1)).flatten()
        py_r = target_scaler.inverse_transform(py.reshape(-1, 1)).flatten()
        fm = float(mean_absolute_error(py_r, py_pred_r))
        final_maes.append(fm)
        print(f"    [naive_ft] final eval on chunk {prev_idx + 1}: MAE={fm:.4f}")

    # BWT scalar: mean of (final_mae_j - diagonal_mae_j) for j in prior chunks.
    # Positive BWT = forgetting (MAE increased after later training) = bad.
    bwt_scores = [final_maes[j] - diagonal_maes[j] for j in range(len(final_maes))]
    naive_ft_bwt = float(np.mean(bwt_scores)) if bwt_scores else float("nan")
    print(f"    [naive_ft] BWT scalar: {naive_ft_bwt:+.4f} (positive = forgetting)")

    return {
        "naive_ft_bwt": naive_ft_bwt,
        "diagonal_maes": diagonal_maes,
        "final_maes": final_maes,
        "bwt_per_chunk": bwt_scores,
    }


# Returns True when a stage should run under the selected run-until cutoff.
def should_run_stage(stage_name: str, run_until: str) -> bool:
    # Validate stage names and compare position in ordered stage sequence.
    if stage_name not in STAGE_ORDER:
        raise ValueError(f"Unknown stage: {stage_name}")
    if run_until not in STAGE_ORDER:
        raise ValueError(
            f"Invalid --run-until value: {run_until}. "
            f"Choose one of: {', '.join(STAGE_ORDER)}"
        )
    return STAGE_ORDER.index(stage_name) <= STAGE_ORDER.index(run_until)


# Builds a stable config signature used to validate reusable artifacts.
def config_signature(cfg: ExperimentConfig) -> str:
    # Only include settings that affect generated artifacts.
    sig_payload = {
        "preprocessed_path": cfg.preprocessed_path,
        "interval": cfg.interval,
        "history_length": cfg.history_length,
        "forecast_steps": cfg.forecast_steps,
        "n_chunks": cfg.n_chunks,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "ewc_lambda": cfg.ewc_lambda,
        "replay_memory": cfg.replay_memory,
        "replay_ratio": cfg.replay_ratio,
        "roll_window": cfg.roll_window,
        "sla_tol": cfg.sla_tol,
        "seed": cfg.seed,
    }
    raw = json.dumps(sig_payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# Loads manifest JSON from disk if present, otherwise returns empty structure.
def load_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / MANIFEST_FILE
    if not manifest_path.exists():
        return {"artifacts": {}, "stages": {}, "signature": None}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Saves manifest JSON to disk.
def save_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    manifest_path = output_dir / MANIFEST_FILE
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# Registers one artifact in manifest with timestamp and optional note.
def mark_artifact(
    manifest: dict[str, Any], name: str, path: Path, note: str = ""
) -> None:
    manifest.setdefault("artifacts", {})[name] = {
        "path": str(path),
        "exists": path.exists(),
        "mtime": path.stat().st_mtime if path.exists() else None,
        "note": note,
    }


# Registers stage status in manifest.
def mark_stage(
    manifest: dict[str, Any], stage: str, status: str, detail: str = ""
) -> None:
    manifest.setdefault("stages", {})[stage] = {
        "status": status,
        "detail": detail,
        "updated_at": time.time(),
    }


# Returns True if all listed files exist.
def artifacts_exist(paths: list[Path]) -> bool:
    return all(p.exists() for p in paths)


# Converts chunk result keys loaded from CSV/JSON to expected numeric types.
def normalize_chunk_results(
    chunk_records: list[dict[str, Any]],
) -> list[dict[str, float]]:
    normalized: list[dict[str, float]] = []
    for r in chunk_records:
        normalized.append(
            {
                "chunk": int(r["chunk"]),
                "mae": float(r["mae"]),
                "rmse": float(r["rmse"]),
                "mape": float(r["mape"]),
            }
        )
    return normalized


# MLP-only model for ablation study comparison.
def create_mlp_only_model(n_static: int) -> Model:
    inp = Input(shape=(n_static,))
    x = Dense(64, activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    out = Dense(1, activation="linear")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# Trains one ablation variant and returns (final_chunk_mae, final_chunk_rmse).
def train_ablation_variant(
    cfg: ExperimentConfig,
    chunks: list[dict[str, np.ndarray]],
    n_temporal: int,
    n_static: int,
    target_scaler: StandardScaler,
    mode: str,
) -> tuple[float, float]:
    import copy
    vcfg = copy.copy(cfg)

    if mode == "hybrid_ewc":
        vcfg.replay_ratio = 0.0
    elif mode == "hybrid_er":
        vcfg.ewc_lambda = 0.0

    if mode in ("hybrid_ewc", "hybrid_er"):
        _, cr, _, _, _, yreal, ypred = train_proposed_model(
            vcfg, chunks, n_temporal, n_static, target_scaler
        )
        mae = float(mean_absolute_error(yreal, ypred))
        rmse = float(np.sqrt(mean_squared_error(yreal, ypred)))
        return mae, rmse

    if mode == "lstm_only":
        model = create_lstm_only_model(cfg.history_length, n_temporal)
        def get_inputs(xseq, xstat):
            return xseq
    elif mode == "mlp_only":
        model = create_mlp_only_model(n_static)
        def get_inputs(xseq, xstat):
            return xstat
    else:  # hybrid_no_cl
        model = create_hybrid_model(cfg.history_length, n_temporal, n_static)
        def get_inputs(xseq, xstat):
            return [xseq, xstat]

    val_sets = []
    for chunk in chunks:
        split = int(0.8 * len(chunk["y"]))
        val_sets.append((chunk["X_seq"][split:], chunk["X_static"][split:], chunk["y"][split:]))
        model.fit(
            get_inputs(chunk["X_seq"][:split], chunk["X_static"][:split]),
            chunk["y"][:split],
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            verbose=0,
        )

    xsv, xstv, yv = val_sets[-1]
    ypred_s = model.predict(get_inputs(xsv, xstv), verbose=0).flatten()
    ypred = target_scaler.inverse_transform(ypred_s.reshape(-1, 1)).flatten()
    yreal = target_scaler.inverse_transform(yv.reshape(-1, 1)).flatten()
    return float(mean_absolute_error(yreal, ypred)), float(np.sqrt(mean_squared_error(yreal, ypred)))


# Runs all 6 ablation variants and returns a summary DataFrame.
def run_ablation_study(
    cfg: ExperimentConfig,
    chunks: list[dict[str, np.ndarray]],
    n_temporal: int,
    n_static: int,
    target_scaler: StandardScaler,
    proposed_mae: float,
    proposed_rmse: float,
) -> pd.DataFrame:
    variants = [
        ("A", "LSTM-only (no MLP)", "lstm_only"),
        ("B", "MLP-only (no LSTM)", "mlp_only"),
        ("C", "Hybrid (no CL)", "hybrid_no_cl"),
        ("D", "Hybrid + EWC only", "hybrid_ewc"),
        ("E", "Hybrid + ER only", "hybrid_er"),
    ]
    rows = []
    for label, desc, mode in variants:
        print(f"    [ablation] variant {label}: {desc}")
        mae, rmse = train_ablation_variant(cfg, chunks, n_temporal, n_static, target_scaler, mode)
        print(f"      MAE={mae:.4f}, RMSE={rmse:.4f}")
        rows.append({"Variant": label, "Description": desc, "MAE": mae, "RMSE": rmse})
    rows.append({"Variant": "F", "Description": "Proposed (EWC+ER) [full]", "MAE": proposed_mae, "RMSE": proposed_rmse})
    return pd.DataFrame(rows)


# Trains proposed model with multiple random seeds and runs paired t-tests vs baselines.
def run_multi_seed_validation(
    cfg: ExperimentConfig,
    chunks: list[dict[str, np.ndarray]],
    n_temporal: int,
    n_static: int,
    target_scaler: StandardScaler,
    output_dir: Path | None = None,
    baseline_results_path: Path | None = None,
) -> pd.DataFrame:
    import copy, math
    seeds = [42, 43, 44, 45, 46]
    rows = []
    for seed in seeds:
        print(f"    [multi_seed] seed={seed}")
        vcfg = copy.copy(cfg)
        vcfg.seed = seed
        configure_environment(seed)
        _, _, _, _, _, yreal, ypred = train_proposed_model(vcfg, chunks, n_temporal, n_static, target_scaler)
        mae = float(mean_absolute_error(yreal, ypred))
        print(f"      MAE={mae:.4f}")
        rows.append({"seed": seed, "mae": mae})
    df = pd.DataFrame(rows)
    mae_vals = df["mae"].values
    mean_mae = float(np.mean(mae_vals))
    std_mae = float(np.std(mae_vals, ddof=1))
    ci_lo, ci_hi = stats.t.interval(0.95, df=len(seeds) - 1, loc=mean_mae, scale=std_mae / math.sqrt(len(seeds)))
    summary = pd.DataFrame([{"seed": "summary", "mae": mean_mae, "std": std_mae, "ci_lo_95": float(ci_lo), "ci_hi_95": float(ci_hi)}])

    # Paired t-test vs each baseline: one-sample t-test with baseline MAE as population mean.
    # Loads baseline MAEs from saved baseline_results.json if available.
    if baseline_results_path is not None and baseline_results_path.exists():
        try:
            with open(baseline_results_path, "r", encoding="utf-8") as f:
                baseline_records = json.load(f)
            sig_rows = []
            for rec in baseline_records:
                bname = rec.get("Method", "")
                if bname == "Proposed (EWC+ER)":
                    continue
                bmae = float(rec.get("MAE", 0))
                t_stat, p_val = stats.ttest_1samp(mae_vals, popmean=bmae)
                cohens_d = (mean_mae - bmae) / std_mae if std_mae > 0 else float("nan")
                sig_rows.append({
                    "comparison": f"Proposed vs {bname}",
                    "proposed_mean_mae": round(mean_mae, 4),
                    "baseline_mae": round(bmae, 4),
                    "t_stat": round(float(t_stat), 4),
                    "p_value": round(float(p_val), 4),
                    "significant_p05": bool(p_val < 0.05),
                    "cohens_d": round(float(cohens_d), 4),
                    "effect_size": (
                        "large" if abs(cohens_d) >= 0.8
                        else "medium" if abs(cohens_d) >= 0.5
                        else "small"
                    ),
                })
                print(f"    [t-test] Proposed vs {bname}: t={t_stat:.4f}, p={p_val:.4f}, d={cohens_d:.4f}")
            if sig_rows and output_dir is not None:
                sig_path = output_dir / "significance_tests.csv"
                pd.DataFrame(sig_rows).to_csv(sig_path, index=False)
                print(f"    [t-test] significance_tests.csv saved -> {sig_path}")
        except Exception as exc:
            print(f"    [t-test] could not compute significance tests: {exc}")

    return pd.concat([df, summary], ignore_index=True)


# Sweeps EWC lambda, replay buffer, and replay ratio to measure sensitivity.
def run_sensitivity_analysis(
    cfg: ExperimentConfig,
    chunks: list[dict[str, np.ndarray]],
    n_temporal: int,
    n_static: int,
    target_scaler: StandardScaler,
) -> dict[str, Any]:
    import copy
    sens_epochs = min(cfg.epochs, 10)
    results: dict[str, list[dict[str, float]]] = {"lambda": [], "buffer": [], "ratio": []}

    for lam in [10.0, 100.0, 500.0, 1000.0, 5000.0]:
        vcfg = copy.copy(cfg)
        vcfg.ewc_lambda = lam
        vcfg.epochs = sens_epochs
        _, _, _, _, _, yreal, ypred = train_proposed_model(vcfg, chunks, n_temporal, n_static, target_scaler)
        mae = float(mean_absolute_error(yreal, ypred))
        print(f"    [sensitivity] lambda={lam} -> MAE={mae:.4f}")
        results["lambda"].append({"value": lam, "mae": mae})

    for buf in [100, 500, 1000, 2000]:
        vcfg = copy.copy(cfg)
        vcfg.replay_memory = buf
        vcfg.epochs = sens_epochs
        _, _, _, _, _, yreal, ypred = train_proposed_model(vcfg, chunks, n_temporal, n_static, target_scaler)
        mae = float(mean_absolute_error(yreal, ypred))
        print(f"    [sensitivity] buffer={buf} -> MAE={mae:.4f}")
        results["buffer"].append({"value": buf, "mae": mae})

    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
        vcfg = copy.copy(cfg)
        vcfg.replay_ratio = ratio
        vcfg.epochs = sens_epochs
        _, _, _, _, _, yreal, ypred = train_proposed_model(vcfg, chunks, n_temporal, n_static, target_scaler)
        mae = float(mean_absolute_error(yreal, ypred))
        print(f"    [sensitivity] ratio={ratio} -> MAE={mae:.4f}")
        results["ratio"].append({"value": ratio, "mae": mae})

    return results


# Evaluates the saved proposed model on holdout apps not seen during training.
# Scalers are passed as objects (already loaded in the calling scope) to avoid extra IO.
def run_cross_app_evaluation(
    model_path: Path,
    ts_df: pd.DataFrame,
    cfg: ExperimentConfig,
    temporal_scaler: StandardScaler,
    static_scaler: StandardScaler,
    target_scaler: StandardScaler,
    temporal_features: list[str],
    static_features: list[str],
    target: str,
) -> dict[str, Any]:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg.seed)
    _, test_idx = next(gss.split(ts_df, groups=ts_df["app_name"]))
    holdout_df = ts_df.iloc[test_idx].copy()

    app_labels = {app: i for i, app in enumerate(ts_df["app_name"].unique())}
    holdout_df["app_id"] = holdout_df["app_name"].map(app_labels).fillna(-1)
    static_features_full = static_features + ["app_id"]

    all_x_seq, all_x_stat, all_y = [], [], []
    min_steps = cfg.history_length + cfg.forecast_steps + 10
    for _, grp in holdout_df.groupby("app_name", sort=False):
        grp = grp.sort_values("timestamp")
        if len(grp) < min_steps:
            continue
        t_arr = grp[temporal_features].values.astype(np.float32)
        s_arr = grp[static_features_full].values.astype(np.float32)
        y_arr = grp[target].values.astype(np.float32)
        x_seq, x_stat, y = create_sequences(t_arr, s_arr, y_arr, cfg.history_length, cfg.forecast_steps)
        all_x_seq.append(x_seq)
        all_x_stat.append(x_stat)
        all_y.append(y)

    if not all_x_seq:
        return {"holdout_rows": int(len(test_idx)), "mae": None, "rmse": None, "sla_viol_pct": None, "note": "no valid sequences"}

    x_seq_all = np.concatenate(all_x_seq, axis=0)
    x_stat_all = np.concatenate(all_x_stat, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    n_t = x_seq_all.shape[2]
    x_seq_s = temporal_scaler.transform(x_seq_all.reshape(-1, n_t)).reshape(x_seq_all.shape)
    x_stat_s = static_scaler.transform(x_stat_all)
    y_s = target_scaler.transform(y_all.reshape(-1, 1)).flatten()

    model = tf.keras.models.load_model(str(model_path))
    y_pred_s = model.predict([x_seq_s, x_stat_s], verbose=0).flatten()
    y_pred = target_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
    y_real = target_scaler.inverse_transform(y_s.reshape(-1, 1)).flatten()

    mae = float(mean_absolute_error(y_real, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_real, y_pred)))
    _, _, sla_viol = compute_provisioning_metrics(y_real, y_pred, cfg.sla_tol)
    print(f"    holdout apps: {holdout_df['app_name'].nunique()}, sequences: {len(y_real):,}")
    print(f"    cross-app MAE={mae:.4f}, RMSE={rmse:.4f}, SLA_Viol%={sla_viol:.2f}")
    return {"holdout_rows": int(len(test_idx)), "holdout_sequences": int(len(y_real)), "mae": mae, "rmse": rmse, "sla_viol_pct": sla_viol}


# Runs the full experiment pipeline and returns a dictionary of key outputs.
def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    # Prepare output folder and load manifest/state metadata.
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(output_dir)

    # Compute selected stages considering run-until and skip flags.
    selected_stages: list[str] = []
    for stage in STAGE_ORDER:
        if not should_run_stage(stage, cfg.run_until):
            continue
        if stage == "multi_seed" and not cfg.run_multi_seed:
            continue
        if stage == "ablation" and not cfg.run_ablation:
            continue
        if stage == "forgetting" and not cfg.run_forgetting:
            continue
        if stage == "naive_ft" and not cfg.run_naive_ft:
            continue
        if stage == "cross_app" and not cfg.run_cross_app:
            continue
        if stage == "sensitivity" and not cfg.run_sensitivity:
            continue
        if stage == "sla" and not cfg.run_sla_analysis:
            continue
        if stage == "dashboard" and not cfg.run_dashboard:
            continue
        selected_stages.append(stage)

    # Compute and compare config signature for safe artifact reuse.
    current_sig = config_signature(cfg)
    prev_sig = manifest.get("signature")
    signature_matches = prev_sig == current_sig
    print(f"Execution plan (run-until={cfg.run_until}): {', '.join(selected_stages)}")
    if prev_sig is None:
        print("Manifest status: no prior signature found (fresh run context)")
    else:
        print(
            f"Manifest status: signature {'MATCH' if signature_matches else 'MISMATCH'}"
        )
    tracker = ProgressTracker(len(selected_stages))

    # Initialize outputs/state holders for partial runs.
    ts_df: pd.DataFrame | None = None
    chunks: list[dict[str, np.ndarray]] | None = None
    temporal_scaler: StandardScaler | None = None
    static_scaler: StandardScaler | None = None
    target_scaler: StandardScaler | None = None
    model: Model | None = None
    chunk_results: list[dict[str, float]] = []
    bwt_matrix: dict[int, dict[int, float]] = {}
    train_histories: list[Any] = []
    val_sets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    y_final_real: np.ndarray | None = None
    y_final_pred: np.ndarray | None = None
    results_df: pd.DataFrame | None = None
    preds: dict[str, np.ndarray] = {}
    baseline_models: dict[str, Model] = {}
    stage_status: dict[str, str] = {}
    baseline_status: dict[str, str] = {}
    n_temporal = 0
    n_static = 0

    # Define feature groups once and reuse across stages.
    temporal_features = [
        "cpu_demand",
        "cpu_diff",
        "cpu_roll_mean",
        "cpu_roll_std",
        "cpu_roll_min",
        "cpu_roll_max",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]
    static_features = [
        "gpu_request_mean",
        "memory_request_mean",
        "rdma_request_mean",
        "role_hn_fraction",
        "max_instance_per_node",
    ]
    target = "cpu_demand"

    # Stage: setup.
    if "setup" in selected_stages:
        s = tracker.start("Configuring environment")
        configure_environment(cfg.seed)
        print(f"    TensorFlow version: {tf.__version__}")
        print(f"    precision policy : {tf.keras.mixed_precision.global_policy().name}")
        print(f"    epochs/batch     : {cfg.epochs}/{cfg.batch_size}")
        stage_status["setup"] = "executed"
        mark_stage(manifest, "setup", "executed", "environment configured")
        tracker.end(s)

    # Stage: load data.
    if "load" in selected_stages:
        s = tracker.start("Loading preprocessed time-series")
        ts_df = load_preprocessed_timeseries(cfg)
        stage_status["load"] = "executed"
        mark_stage(manifest, "load", "executed", "preprocessed data loaded")
        tracker.end(s)

    # Stage: feature engineering.
    if "features" in selected_stages:
        if ts_df is None:
            raise RuntimeError(
                "Stage dependency missing: load must run before features."
            )
        s = tracker.start("Engineering temporal and static features")
        ts_df = engineer_features(ts_df, cfg.roll_window)
        stage_status["features"] = "executed"
        mark_stage(manifest, "features", "executed", "feature engineering complete")
        tracker.end(
            s,
            note=f"features ready (temporal={len(temporal_features)}, static={len(static_features)})",
        )

    # Stage: preparation.
    if "prepare" in selected_stages:
        if ts_df is None:
            raise RuntimeError(
                "Stage dependency missing: features must run before prepare."
            )
        s = tracker.start("Preparing sequences, scalers, and chunks")
        chunks, temporal_scaler, static_scaler, target_scaler, n_temporal, n_static = prepare_model_inputs(
            ts_df,
            cfg,
            temporal_features,
            static_features,
            target,
            output_dir,
        )
        stage_status["prepare"] = "executed"
        mark_stage(manifest, "prepare", "executed", "sequences/chunks prepared")
        tracker.end(s)

    # Stage: proposed model (with resume/reuse support).
    if "train" in selected_stages:
        if chunks is None or target_scaler is None:
            raise RuntimeError(
                "Stage dependency missing: prepare must run before train."
            )
        s = tracker.start("Training proposed model (Hybrid + EWC + ER)")

        # Define proposed-stage artifact set used for safe reuse.
        proposed_model_path = output_dir / "hybrid_model_ewc_er.keras"
        chunk_metrics_path = output_dir / "chunk_metrics.csv"
        bwt_matrix_path = output_dir / "bwt_matrix.json"
        train_histories_path = output_dir / "train_histories.pkl"
        final_pred_path = output_dir / "final_predictions.npz"
        proposed_artifacts = [
            proposed_model_path,
            chunk_metrics_path,
            bwt_matrix_path,
            train_histories_path,
            final_pred_path,
        ]

        # Decide whether to reuse artifacts or retrain, with detailed status logs.
        can_resume_proposed = (
            cfg.resume
            and signature_matches
            and (not cfg.force_retrain)
            and (not cfg.refresh_all)
            and artifacts_exist(proposed_artifacts)
        )

        if can_resume_proposed:
            print(
                "    [train][proposed] all saved artifacts found -> loading and skipping training"
            )
            model = tf.keras.models.load_model(proposed_model_path)
            chunk_df = pd.read_csv(chunk_metrics_path)
            chunk_results = normalize_chunk_results(chunk_df.to_dict(orient="records"))
            with open(bwt_matrix_path, "r", encoding="utf-8") as f:
                bwt_raw = json.load(f)
            bwt_matrix = {
                int(k): {int(kk): float(vv) for kk, vv in v.items()}
                for k, v in bwt_raw.items()
            }
            with open(train_histories_path, "rb") as f:
                train_histories = pickle.load(f)
            pred_npz = np.load(final_pred_path)
            y_final_real = pred_npz["y_final_real"]
            y_final_pred = pred_npz["y_final_pred"]
            val_sets = build_val_sets_from_chunks(chunks)
            stage_status["train"] = "loaded"
            mark_stage(
                manifest, "train", "loaded", "reused saved proposed model pipeline"
            )
            tracker.end(s, note="loaded cached proposed pipeline")
        else:
            print("    [train][proposed] cache not usable -> training required")
            (
                model,
                chunk_results,
                bwt_matrix,
                train_histories,
                val_sets,
                y_final_real,
                y_final_pred,
            ) = train_proposed_model(
                cfg,
                chunks,
                n_temporal,
                n_static,
                target_scaler,
            )

            # Persist proposed-stage artifacts for future resume runs.
            model.save(proposed_model_path)
            pd.DataFrame(chunk_results).to_csv(chunk_metrics_path, index=False)
            with open(bwt_matrix_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        str(k): {str(kk): vv for kk, vv in v.items()}
                        for k, v in bwt_matrix.items()
                    },
                    f,
                    indent=2,
                )
            with open(train_histories_path, "wb") as f:
                pickle.dump(train_histories, f)
            np.savez(
                final_pred_path, y_final_real=y_final_real, y_final_pred=y_final_pred
            )

            mark_artifact(manifest, "proposed_model", proposed_model_path)
            mark_artifact(manifest, "chunk_metrics", chunk_metrics_path)
            mark_artifact(manifest, "bwt_matrix", bwt_matrix_path)
            mark_artifact(manifest, "train_histories", train_histories_path)
            mark_artifact(manifest, "final_predictions", final_pred_path)
            stage_status["train"] = "trained"
            mark_stage(
                manifest,
                "train",
                "trained",
                "proposed model pipeline executed and saved",
            )
            tracker.end(s, note="trained and saved proposed pipeline")

    # Stage: baselines with per-baseline cache support.
    if "baselines" in selected_stages:
        if (
            chunks is None
            or target_scaler is None
            or y_final_real is None
            or y_final_pred is None
        ):
            raise RuntimeError(
                "Stage dependency missing: train must run before baselines."
            )
        if not val_sets:
            val_sets = build_val_sets_from_chunks(chunks)
        s = tracker.start("Running baseline models and comparisons")
        x_final_seq, x_final_stat, _ = val_sets[-1]
        results_df, preds, baseline_models, baseline_status = run_baselines_with_cache(
            cfg,
            chunks,
            n_temporal,
            n_static,
            x_final_seq,
            x_final_stat,
            y_final_real,
            y_final_pred,
            target_scaler,
            output_dir,
            manifest,
            signature_matches,
        )
        print("    baseline statuses:")
        for k, v in baseline_status.items():
            print(f"      {k:<18} -> {v}")
        print(results_df.round(4).to_string())
        stage_status["baselines"] = "executed"
        mark_stage(
            manifest,
            "baselines",
            "executed",
            "baseline models completed with cache checks",
        )
        tracker.end(s)

    # Stage: core outputs with reusable table/plot artifacts.
    if "core_outputs" in selected_stages:
        if results_df is None or y_final_real is None:
            raise RuntimeError(
                "Stage dependency missing: baselines must run before core_outputs."
            )
        s = tracker.start("Saving result tables and core figures")
        save_tables(output_dir, chunk_results, results_df)
        plot_evaluation_results(
            output_dir,
            chunk_results,
            bwt_matrix,
            y_final_real,
            preds,
            results_df,
            train_histories,
        )
        mark_artifact(
            manifest, "baseline_results_csv", output_dir / "baseline_results.csv"
        )
        mark_artifact(manifest, "chunk_metrics_csv", output_dir / "chunk_metrics.csv")
        mark_artifact(
            manifest, "evaluation_results_png", output_dir / "evaluation_results.png"
        )
        stage_status["core_outputs"] = "executed"
        mark_stage(
            manifest, "core_outputs", "executed", "core tables and figure written"
        )
        tracker.end(
            s, note="saved chunk_metrics/baseline_results + evaluation_results.png"
        )

    # Stage: summary.
    if "summary" in selected_stages:
        if y_final_real is None or y_final_pred is None:
            raise RuntimeError(
                "Stage dependency missing: train must run before summary."
            )
        s = tracker.start("Printing core metrics summary")
        over_p, under_p, sla = compute_provisioning_metrics(y_final_real, y_final_pred)
        print("    per-chunk metrics:")
        for r in chunk_results:
            print(
                f"      chunk {r['chunk']}: MAE={r['mae']:.4f}, RMSE={r['rmse']:.4f}, MAPE={r['mape']:.2f}%"
            )
        print(f"    final over-provisioning : {over_p:.2f}")
        print(f"    final under-provisioning: {under_p:.2f}")
        print(f"    final SLA violation (%) : {sla:.2f}")
        stage_status["summary"] = "executed"
        mark_stage(manifest, "summary", "executed", "terminal summary printed")
        tracker.end(s)

    # Stage: multi-seed validation — trains with 5 seeds for CI and reproducibility.
    if "multi_seed" in selected_stages:
        if chunks is None or target_scaler is None:
            raise RuntimeError("Stage dependency missing: prepare must run before multi_seed.")
        s = tracker.start("Validation: Multi-seed statistical testing")
        ms_file = output_dir / "multi_seed_stats.csv"
        if (
            cfg.resume
            and signature_matches
            and (not cfg.force_validation and not cfg.refresh_all)
            and ms_file.exists()
        ):
            print("    [validation][multi_seed] found multi_seed_stats.csv -> skipping")
            stage_status["multi_seed"] = "loaded"
            mark_stage(
                manifest, "multi_seed", "loaded", "reused existing multi-seed results"
            )
            tracker.end(s, note="loaded")
        else:
            ms_df = run_multi_seed_validation(
                cfg, chunks, n_temporal, n_static, target_scaler,
                output_dir=output_dir,
                baseline_results_path=output_dir / "baseline_results.json",
            )
            ms_df.to_csv(ms_file, index=False)
            mark_artifact(manifest, "multi_seed_stats", ms_file)
            stage_status["multi_seed"] = "executed"
            mark_stage(manifest, "multi_seed", "executed", "multi-seed validation complete")
            tracker.end(s, note="executed")

    # Stage: ablation study — trains 6 variants and compares MAE.
    if "ablation" in selected_stages:
        if chunks is None or target_scaler is None:
            raise RuntimeError("Stage dependency missing: prepare must run before ablation.")
        s = tracker.start("Validation: Ablation study")
        abl_file = output_dir / "ablation_results.csv"
        if (
            cfg.resume
            and signature_matches
            and (not cfg.force_validation and not cfg.refresh_all)
            and abl_file.exists()
        ):
            print("    [validation][ablation] found ablation_results.csv -> skipping")
            stage_status["ablation"] = "loaded"
            mark_stage(
                manifest, "ablation", "loaded", "reused existing ablation results"
            )
            tracker.end(s, note="loaded")
        else:
            p_mae = chunk_results[-1]["mae"] if chunk_results else 0.0
            p_rmse = chunk_results[-1]["rmse"] if chunk_results else 0.0
            abl_df = run_ablation_study(cfg, chunks, n_temporal, n_static, target_scaler, p_mae, p_rmse)
            abl_df.to_csv(abl_file, index=False)
            mark_artifact(manifest, "ablation_results", abl_file)
            stage_status["ablation"] = "executed"
            mark_stage(manifest, "ablation", "executed", "ablation study complete")
            tracker.end(s, note="executed")

    # Stage: forgetting scalar with optional cache.
    if "forgetting" in selected_stages:
        if not chunk_results:
            raise RuntimeError(
                "Stage dependency missing: train must run before forgetting."
            )
        s = tracker.start("Validation: Forgetting (BWT deep dive)")
        bwt_file = output_dir / "forgetting_results.json"
        if (
            cfg.resume
            and signature_matches
            and (not cfg.force_validation and not cfg.refresh_all)
            and bwt_file.exists()
        ):
            print(
                "    [validation][forgetting] found forgetting_results.json -> loading"
            )
            with open(bwt_file, "r", encoding="utf-8") as f:
                bwt_payload = json.load(f)
            bwt_scalar = float(bwt_payload.get("bwt_scalar", 0.0))
            print(f"    Proposed BWT scalar: {bwt_scalar:+.6f}")
            stage_status["forgetting"] = "loaded"
            mark_stage(manifest, "forgetting", "loaded", "reused forgetting scalar")
            tracker.end(s, note="loaded")
        else:
            bwt_scalar = compute_bwt_scalar(bwt_matrix, chunk_results)
            print(f"    Proposed BWT scalar: {bwt_scalar:+.6f}")
            with open(bwt_file, "w", encoding="utf-8") as f:
                json.dump({"bwt_scalar": bwt_scalar}, f, indent=2)
            mark_artifact(manifest, "forgetting_results", bwt_file)
            stage_status["forgetting"] = "computed"
            mark_stage(
                manifest,
                "forgetting",
                "computed",
                "computed and saved forgetting scalar",
            )
            tracker.end(s, note="computed")

    # Stage: naive fine-tuning BWT comparison — trains sequential model without EWC/ER.
    if "naive_ft" in selected_stages:
        if chunks is None or target_scaler is None:
            raise RuntimeError(
                "Stage dependency missing: prepare must run before naive_ft."
            )
        s = tracker.start("Validation: Naive FT BWT comparison")
        nft_file = output_dir / "naive_ft_results.json"
        if (
            cfg.resume
            and signature_matches
            and (not cfg.force_validation and not cfg.refresh_all)
            and nft_file.exists()
        ):
            print("    [validation][naive_ft] found naive_ft_results.json -> skipping")
            stage_status["naive_ft"] = "loaded"
            mark_stage(manifest, "naive_ft", "loaded", "reused naive FT results")
            tracker.end(s, note="loaded")
        else:
            nft_result = run_naive_ft_comparison(
                cfg, chunks, n_temporal, n_static, target_scaler
            )
            with open(nft_file, "w", encoding="utf-8") as f:
                json.dump(nft_result, f, indent=2)
            mark_artifact(manifest, "naive_ft_results", nft_file)
            stage_status["naive_ft"] = "executed"
            mark_stage(manifest, "naive_ft", "executed", "naive FT BWT comparison complete")
            tracker.end(s, note=f"executed — naive_ft_bwt={nft_result['naive_ft_bwt']:+.4f}")

    # Stage: cross-app placeholder with artifact check.
    if "cross_app" in selected_stages:
        if ts_df is None or temporal_scaler is None or static_scaler is None:
            raise RuntimeError(
                "Stage dependency missing: load/features/prepare must run before cross_app."
            )
        s = tracker.start("Validation: Cross-app generalization")
        cross_file = output_dir / "cross_app_results.json"
        if (
            cfg.resume
            and signature_matches
            and (not cfg.force_validation and not cfg.refresh_all)
            and cross_file.exists()
        ):
            print(
                "    [validation][cross_app] found cross_app_results.json -> skipping"
            )
            stage_status["cross_app"] = "loaded"
            mark_stage(
                manifest, "cross_app", "loaded", "reused existing cross-app results"
            )
            tracker.end(s, note="loaded")
        else:
            proposed_model_path = output_dir / "hybrid_model_ewc_er.keras"
            cross_result = run_cross_app_evaluation(
                proposed_model_path, ts_df, cfg,
                temporal_scaler, static_scaler, target_scaler,
                temporal_features, static_features, target,
            )
            with open(cross_file, "w", encoding="utf-8") as f:
                json.dump(cross_result, f, indent=2)
            mark_artifact(manifest, "cross_app_results", cross_file)
            stage_status["cross_app"] = "executed"
            mark_stage(manifest, "cross_app", "executed", "cross-app evaluation complete")
            tracker.end(s, note="executed")

    # Stage: hyperparameter sensitivity — sweeps lambda, buffer, ratio one-at-a-time.
    if "sensitivity" in selected_stages:
        if chunks is None or target_scaler is None:
            raise RuntimeError("Stage dependency missing: prepare must run before sensitivity.")
        s = tracker.start("Validation: Hyperparameter sensitivity")
        sens_file = output_dir / "sensitivity_results.json"
        if (
            cfg.resume
            and signature_matches
            and (not cfg.force_validation and not cfg.refresh_all)
            and sens_file.exists()
        ):
            print(
                "    [validation][sensitivity] found sensitivity_results.json -> skipping"
            )
            stage_status["sensitivity"] = "loaded"
            mark_stage(manifest, "sensitivity", "loaded", "reused sensitivity results")
            tracker.end(s, note="loaded")
        else:
            sens_results = run_sensitivity_analysis(cfg, chunks, n_temporal, n_static, target_scaler)
            with open(sens_file, "w", encoding="utf-8") as f:
                json.dump(sens_results, f, indent=2)
            mark_artifact(manifest, "sensitivity_results", sens_file)
            stage_status["sensitivity"] = "executed"
            mark_stage(manifest, "sensitivity", "executed", "sensitivity analysis complete")
            tracker.end(s, note="executed")

    # Stage: SLA placeholder with artifact check.
    if "sla" in selected_stages:
        s = tracker.start("Validation: SLA compliance and cost analysis")
        sla_file = output_dir / "sla_results.csv"
        if (
            cfg.resume
            and signature_matches
            and (not cfg.force_validation and not cfg.refresh_all)
            and sla_file.exists()
        ):
            print("    [validation][sla] found sla_results.csv -> skipping")
            stage_status["sla"] = "loaded"
            mark_stage(manifest, "sla", "loaded", "reused SLA results")
            tracker.end(s, note="loaded")
        else:
            print("    SLA table already included in baseline results columns.")
            if results_df is not None:
                sla_table = results_df[["SLA_Viol%", "OverProv%", "UnderProv%"]].copy()
                sla_table.to_csv(sla_file)
                mark_artifact(manifest, "sla_results", sla_file)
            stage_status["sla"] = "completed"
            mark_stage(manifest, "sla", "completed", "SLA summary exported")
            tracker.end(s, note="completed")

    # Stage: dashboard with cache check.
    if "dashboard" in selected_stages:
        if results_df is None:
            raise RuntimeError(
                "Stage dependency missing: baselines must run before dashboard."
            )
        s = tracker.start("Validation: Final dashboard assembly")
        dashboard_path = output_dir / "validation_dashboard.png"
        if (
            cfg.resume
            and signature_matches
            and (not cfg.force_validation and not cfg.refresh_all)
            and dashboard_path.exists()
        ):
            print(
                "    [validation][dashboard] found validation_dashboard.png -> skipping"
            )
            stage_status["dashboard"] = "loaded"
            mark_stage(manifest, "dashboard", "loaded", "reused dashboard figure")
            tracker.end(s, note="loaded")
        else:
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(1, 1, 1)
            ax.axis("off")
            best_method = results_df["MAE"].idxmin()
            text = (
                "Validation Dashboard Summary\n\n"
                f"Best method by MAE: {best_method}\n"
                f"Proposed MAE: {results_df.loc['Proposed (EWC+ER)', 'MAE']:.4f}\n"
                f"Proposed RMSE: {results_df.loc['Proposed (EWC+ER)', 'RMSE']:.4f}\n"
                f"Proposed MAPE% (filtered): {results_df.loc['Proposed (EWC+ER)', 'MAPE% (filtered)']:.2f}\n"
                f"Proposed SLA_Viol%: {results_df.loc['Proposed (EWC+ER)', 'SLA_Viol%']:.2f}"
            )
            ax.text(0.02, 0.95, text, va="top", fontsize=11)
            plt.tight_layout()
            plt.savefig(dashboard_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            mark_artifact(manifest, "validation_dashboard", dashboard_path)
            stage_status["dashboard"] = "generated"
            mark_stage(manifest, "dashboard", "generated", "dashboard figure generated")
            tracker.end(s, note="saved validation_dashboard.png")

    # Persist manifest signature and stage status for next resume runs.
    manifest["signature"] = current_sig
    manifest["last_run_until"] = cfg.run_until
    manifest["stage_status"] = stage_status
    save_manifest(output_dir, manifest)

    # Print global runtime and return available outputs.
    tracker.finish()
    return {
        "ts_df": ts_df,
        "chunks": chunks,
        "target_scaler": target_scaler,
        "model": model,
        "chunk_results": chunk_results,
        "bwt_matrix": bwt_matrix,
        "results_df": results_df,
        "preds": preds,
        "y_final_real": y_final_real,
        "y_final_pred": y_final_pred,
        "baseline_models": baseline_models,
        "baseline_status": baseline_status,
        "stage_status": stage_status,
        "run_until": cfg.run_until,
        "executed_stages": selected_stages,
        "manifest": manifest,
    }


# Parses CLI arguments into ExperimentConfig for terminal execution.
def parse_args() -> ExperimentConfig:
    # Define command-line options with defaults aligned to user-selected option 1.
    parser = argparse.ArgumentParser(
        description="Run research implementation with full validation flow"
    )
    parser.add_argument("--preprocessed-path", default="alibaba_timeseries_full.csv")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-until",
        default="dashboard",
        choices=STAGE_ORDER,
        help=("Run pipeline up to this stage. Stages: " + ", ".join(STAGE_ORDER)),
    )
    parser.add_argument("--skip-multi-seed", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-forgetting", action="store_true")
    parser.add_argument("--skip-naive-ft", action="store_true")
    parser.add_argument("--skip-cross-app", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-sla", action="store_true")
    parser.add_argument("--skip-dashboard", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--force-baselines", action="store_true")
    parser.add_argument("--force-validation", action="store_true")
    parser.add_argument("--refresh-all", action="store_true")
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Enable lighter settings where applicable",
    )
    args = parser.parse_args()

    # Construct config object from parsed arguments.
    return ExperimentConfig(
        preprocessed_path=args.preprocessed_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        run_until=args.run_until,
        run_multi_seed=not args.skip_multi_seed,
        run_ablation=not args.skip_ablation,
        run_forgetting=not args.skip_forgetting,
        run_naive_ft=not args.skip_naive_ft,
        run_cross_app=not args.skip_cross_app,
        run_sensitivity=not args.skip_sensitivity,
        run_sla_analysis=not args.skip_sla,
        run_dashboard=not args.skip_dashboard,
        resume=not args.no_resume,
        force_retrain=args.force_retrain,
        force_baselines=args.force_baselines,
        force_validation=args.force_validation,
        refresh_all=args.refresh_all,
        fast_mode=args.fast_mode,
    )


# Script entrypoint that executes the full experiment and reports completion.
def main() -> None:
    # Parse runtime config and execute the complete research pipeline.
    cfg = parse_args()
    print("Selected execution options:")
    print(f"  preprocessed-path : {cfg.preprocessed_path}")
    print(f"  output-dir        : {cfg.output_dir}")
    print(f"  run-until         : {cfg.run_until}")
    print(f"  skip multi-seed   : {not cfg.run_multi_seed}")
    print(f"  skip ablation     : {not cfg.run_ablation}")
    print(f"  skip forgetting   : {not cfg.run_forgetting}")
    print(f"  skip naive-ft     : {not cfg.run_naive_ft}")
    print(f"  skip cross-app    : {not cfg.run_cross_app}")
    print(f"  skip sensitivity  : {not cfg.run_sensitivity}")
    print(f"  skip sla          : {not cfg.run_sla_analysis}")
    print(f"  skip dashboard    : {not cfg.run_dashboard}")
    print(f"  resume            : {cfg.resume}")
    print(f"  force retrain     : {cfg.force_retrain}")
    print(f"  force baselines   : {cfg.force_baselines}")
    print(f"  force validation  : {cfg.force_validation}")
    print(f"  refresh all       : {cfg.refresh_all}")
    run_experiment(cfg)


# Standard Python module guard for CLI execution.
if __name__ == "__main__":
    main()
