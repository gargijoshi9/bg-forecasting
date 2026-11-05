#!/usr/bin/env python3
"""
run_experiments.py
==================
Executes benchmark forecasting experiments on continuous glucose data:
1. Computes sequence generation audits across (60, 120, 360) lookbacks and (30, 60, 120) horizons.
2. Performs chronological segment-aligned temporal partitioning (leakage-free by gap isolation).
3. Evaluates Persistence Baseline and Ridge Regression across all horizons and lookbacks.
4. Outputs documented metrics (MAE, RMSE, R2, Clinical Accuracy) to results/forecasting_experiments.csv.
"""

import os
import pandas as pd
import numpy as np

from src.forecasting.dataset import ForecastingDataset
from src.forecasting.sequence_builder import SequenceBuilder
from src.forecasting.baseline import PersistenceBaseline
from src.forecasting.ridge_model import RidgeForecaster
from src.forecasting.evaluation import Evaluator
from src.forecasting.experiment_tracker import ExperimentTracker


def run_all_benchmarks():
    print("=" * 70)
    print("RUNNING BG FORECASTING INFRASTRUCTURE & EXPERIMENTS")
    print("=" * 70)

    # 1. Load dataset
    ds = ForecastingDataset("data/processed/glucose_features.csv")
    df = ds.load()
    print(f"Loaded {len(df):,} total 5-minute CGM records.")

    # 2. Sequence generation audit across all lookback and horizon combinations
    lookbacks = [60, 120, 360]  # minutes
    horizons = [30, 60, 120]   # minutes
    
    print("\n" + "=" * 70)
    print("SEQUENCE GENERATION AUDIT (ALL COMBINATIONS)")
    print("=" * 70)
    audit_results = []
    for L in lookbacks:
        for H in horizons:
            builder = SequenceBuilder(lookback_minutes=L, horizon_minutes=H)
            seq_ds = builder.build_sequences(df, feature_cols=["glucose"])
            stats = seq_ds.audit_stats
            audit_results.append({
                "lookback_mins": L,
                "horizon_mins": H,
                "lookback_steps": stats["lookback_steps"],
                "horizon_steps": stats["horizon_steps"],
                "total_window_steps": stats["total_window_steps"],
                "theoretical_sliding_windows": stats["theoretical_sliding_windows"],
                "valid_sequences": stats["valid_sequences_count"],
                "excluded_gap_windows": stats["excluded_gap_windows"],
            })
            print(
                f"L={L:3d}m ({stats['lookback_steps']:2d} steps), "
                f"H={H:3d}m ({stats['horizon_steps']:2d} steps) | "
                f"Valid: {stats['valid_sequences_count']:,} | "
                f"Excluded Gaps: {stats['excluded_gap_windows']:3d} | "
                f"Total Possible: {stats['theoretical_sliding_windows']:,}"
            )

    df_audit = pd.DataFrame(audit_results)
    df_audit.to_csv("results/sequence_generation_audit.csv", index=False)
    print("\nSaved audit results to results/sequence_generation_audit.csv")

    # 3. Create chronological temporal split
    # Natural segment split: Segments 1-6 (Train), Segment 7 (Val), Segment 8 (Test)
    # The inter-split gaps are 21h 55m and 38h 15m, strictly preventing window leakage.
    print("\n" + "=" * 70)
    print("CHRONOLOGICAL TEMPORAL PARTITIONING")
    print("=" * 70)
    split = ds.create_segment_split(
        train_segments=[1, 2, 3, 4, 5, 6],
        val_segments=[7],
        test_segments=[8],
    )
    b = split.boundaries
    print(f"Train Partition : {b['train_rows']:,} rows ({b['train_fraction']*100:.1f}%) | {b['train_start']} -> {b['train_end']}")
    print(f"Val Partition   : {b['val_rows']:,} rows ({b['val_fraction']*100:.1f}%) | {b['val_start']} -> {b['val_end']}")
    print(f"Test Partition  : {b['test_rows']:,} rows ({b['test_fraction']*100:.1f}%) | {b['test_start']} -> {b['test_end']}")
    print(f"Train->Val Gap  : {b['inter_split_gap_train_val']}")
    print(f"Val->Test Gap   : {b['inter_split_gap_val_test']}")

    # 4. Feature scaling: Fit on Train partition ONLY
    feature_cols = [
        "glucose",
        "glucose_min",
        "glucose_max",
        "glucose_std",
        "time_sin",
        "time_cos",
        "day_of_week",
        "is_weekend",
        "cycle_day",
        "cycle_length_days",
        "days_to_next_period",
        "is_menstrual_flow",
        "is_stressed",
        "stress_intensity",
    ]
    ds.fit_scaler(split.train_df, feature_cols=feature_cols)
    print(f"\nFitted StandardScaler on {len(split.train_df):,} training rows across {len(feature_cols)} features.")

    train_df_scaled = ds.transform_df(split.train_df, feature_cols=feature_cols)
    val_df_scaled = ds.transform_df(split.val_df, feature_cols=feature_cols)
    test_df_scaled = ds.transform_df(split.test_df, feature_cols=feature_cols)

    # 5. Model Benchmarking
    tracker = ExperimentTracker("results")

    print("\n" + "=" * 70)
    print("EVALUATING PERSISTENCE & RIDGE MODELS")
    print("=" * 70)

    for L in lookbacks:
        for H in horizons:
            builder = SequenceBuilder(
                lookback_minutes=L,
                horizon_minutes=H,
                target_col="glucose",
                timestamp_col="timestamp",
            )

            # Unscaled sequences for persistence (and ground-truth targets)
            train_seq_raw = builder.build_sequences(split.train_df, feature_cols=["glucose"])
            val_seq_raw = builder.build_sequences(split.val_df, feature_cols=["glucose"])
            test_seq_raw = builder.build_sequences(split.test_df, feature_cols=["glucose"])

            # Scaled sequences for Ridge
            train_seq_scaled = builder.build_sequences(train_df_scaled, feature_cols=feature_cols)
            val_seq_scaled = builder.build_sequences(val_df_scaled, feature_cols=feature_cols)
            test_seq_scaled = builder.build_sequences(test_df_scaled, feature_cols=feature_cols)

            # --- MODEL 1: PERSISTENCE BASELINE ---
            p_model = PersistenceBaseline(glucose_idx=0)
            
            # Predict and evaluate on Validation
            p_val_preds = p_model.predict(val_seq_raw.X)
            p_val_metrics = Evaluator.evaluate(
                y_true=val_seq_raw.y,
                y_pred=p_val_preds,
                model_name="Persistence",
                lookback_minutes=L,
                horizon_minutes=H,
            )
            tracker.log_run(p_val_metrics, split_name="validation")

            # Predict and evaluate on Test
            p_test_preds = p_model.predict(test_seq_raw.X)
            p_test_metrics = Evaluator.evaluate(
                y_true=test_seq_raw.y,
                y_pred=p_test_preds,
                model_name="Persistence",
                lookback_minutes=L,
                horizon_minutes=H,
            )
            tracker.log_run(p_test_metrics, split_name="test")

            # --- MODEL 2: RIDGE REGRESSION ---
            ridge = RidgeForecaster()
            # Tune alpha on validation set
            best_alpha = ridge.tune_alpha(
                X_train=train_seq_scaled.X,
                y_train=train_seq_raw.y,  # target in original mg/dL scale
                X_val=val_seq_scaled.X,
                y_val=val_seq_raw.y,
                alpha_grid=[0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0],
            )

            # Evaluate on Validation
            r_val_preds = ridge.predict(val_seq_scaled.X)
            r_val_metrics = Evaluator.evaluate(
                y_true=val_seq_raw.y,
                y_pred=r_val_preds,
                model_name="Ridge",
                lookback_minutes=L,
                horizon_minutes=H,
            )
            tracker.log_run(r_val_metrics, split_name="validation", hyperparams={"alpha": best_alpha})

            # Evaluate on Test
            r_test_preds = ridge.predict(test_seq_scaled.X)
            r_test_metrics = Evaluator.evaluate(
                y_true=test_seq_raw.y,
                y_pred=r_test_preds,
                model_name="Ridge",
                lookback_minutes=L,
                horizon_minutes=H,
            )
            tracker.log_run(r_test_metrics, split_name="test", hyperparams={"alpha": best_alpha})

            print(
                f"[L={L:3d}m, H={H:3d}m] | "
                f"Persistence Test MAE: {p_test_metrics.mae:5.2f} mg/dL, RMSE: {p_test_metrics.rmse:5.2f}, R2: {p_test_metrics.r2:6.3f} | "
                f"Ridge Test MAE: {r_test_metrics.mae:5.2f} mg/dL, RMSE: {r_test_metrics.rmse:5.2f}, R2: {r_test_metrics.r2:6.3f} (alpha={best_alpha})"
            )

    # Save summary
    out_csv = tracker.save("forecasting_experiments.csv")
    print(f"\nAll benchmark experiments complete! Results saved to {out_csv}")

    print("\n" + "=" * 70)
    print("TEST SET BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(tracker.generate_markdown_summary(split="test"))


if __name__ == "__main__":
    run_all_benchmarks()

