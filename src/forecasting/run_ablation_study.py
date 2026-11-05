#!/usr/bin/env python3
"""
run_ablation_study.py
=====================
Systematic ablation study testing whether cycle, stress, and wearable features
improve blood glucose forecasting across multiple horizons (30, 60, 120 mins).
Models evaluated:
1. Persistence Baseline
2. Ridge Regression
3. XGBoost Regressor

Feature Sets:
- FS1: Pure Autoregressive (glucose dynamics only)
- FS2: Autoregressive + Diurnal / Time
- FS3: Autoregressive + Time + Menstrual Cycle
- FS4: Autoregressive + Time + Cycle + Stress
- FS5: Full Multi-Modal (Set 4 + Wearable HR/HRV)
"""

import os
import pandas as pd
import numpy as np

from src.forecasting.dataset import ForecastingDataset
from src.forecasting.sequence_builder import SequenceBuilder
from src.forecasting.baseline import PersistenceBaseline
from src.forecasting.ridge_model import RidgeForecaster
from src.forecasting.xgboost_model import XGBoostForecaster
from src.forecasting.evaluation import Evaluator
from src.forecasting.experiment_tracker import ExperimentTracker


def load_dataset_with_wearables():
    df_wear = pd.read_csv("data/processed/wearable_metrics.csv")
    df = pd.read_csv("data/processed/glucose_features.csv")

    df_wear["dt"] = pd.to_datetime(df_wear["timestamp"])
    df["dt"] = pd.to_datetime(df["timestamp"])

    hr = df_wear[df_wear["metric_type"] == "heart_rate"].sort_values("dt")
    hrv = df_wear[df_wear["metric_type"] == "hrv_sdnn"].sort_values("dt")

    # Merge backward strictly preventing lookahead leakage
    df = pd.merge_asof(
        df,
        hr[["dt", "value"]].rename(columns={"value": "heart_rate_bpm"}),
        on="dt",
        direction="backward",
        tolerance=pd.Timedelta("30min"),
    )
    df = pd.merge_asof(
        df,
        hrv[["dt", "value"]].rename(columns={"value": "hrv_sdnn_ms"}),
        on="dt",
        direction="backward",
        tolerance=pd.Timedelta("60min"),
    )
    df.drop(columns=["dt"], inplace=True)
    return df


def run_ablation():
    print("=" * 75)
    print("RUNNING FEATURE ABLATION STUDY: CYCLE, STRESS & WEARABLE FEATURES")
    print("=" * 75)

    df = load_dataset_with_wearables()
    ds = ForecastingDataset()
    ds.df = df

    # Same split partitions: Segments 1-6 (Train), Segment 7 (Val), Segment 8 (Test)
    split = ds.create_segment_split(
        train_segments=[1, 2, 3, 4, 5, 6],
        val_segments=[7],
        test_segments=[8],
    )

    feature_sets = {
        "FS1_Glucose_Only": ["glucose", "glucose_min", "glucose_max", "glucose_std"],
        "FS2_Glucose_Time": [
            "glucose", "glucose_min", "glucose_max", "glucose_std",
            "time_sin", "time_cos", "day_of_week", "is_weekend"
        ],
        "FS3_Glucose_Time_Cycle": [
            "glucose", "glucose_min", "glucose_max", "glucose_std",
            "time_sin", "time_cos", "day_of_week", "is_weekend",
            "cycle_day", "cycle_length_days", "days_to_next_period", "is_menstrual_flow"
        ],
        "FS4_Glucose_Time_Cycle_Stress": [
            "glucose", "glucose_min", "glucose_max", "glucose_std",
            "time_sin", "time_cos", "day_of_week", "is_weekend",
            "cycle_day", "cycle_length_days", "days_to_next_period", "is_menstrual_flow",
            "is_stressed", "stress_intensity"
        ],
        "FS5_Full_MultiModal": [
            "glucose", "glucose_min", "glucose_max", "glucose_std",
            "time_sin", "time_cos", "day_of_week", "is_weekend",
            "cycle_day", "cycle_length_days", "days_to_next_period", "is_menstrual_flow",
            "is_stressed", "stress_intensity",
            "heart_rate_bpm", "hrv_sdnn_ms"
        ],
    }

    lookback_minutes = 60
    horizons = [30, 60, 120]  # minutes

    records = []

    for H in horizons:
        print(f"\n>>> FORECAST HORIZON: {H} MINUTES (Lookback: {lookback_minutes} mins) <<<")
        builder = SequenceBuilder(lookback_minutes=lookback_minutes, horizon_minutes=H)

        # Baseline: Persistence
        train_base = builder.build_sequences(split.train_df, feature_cols=["glucose"])
        val_base = builder.build_sequences(split.val_df, feature_cols=["glucose"])
        test_base = builder.build_sequences(split.test_df, feature_cols=["glucose"])

        p_model = PersistenceBaseline(glucose_idx=0)
        p_preds = p_model.predict(test_base.X)
        p_metrics = Evaluator.evaluate(
            y_true=test_base.y,
            y_pred=p_preds,
            model_name="Persistence",
            lookback_minutes=lookback_minutes,
            horizon_minutes=H,
        )
        p_row = p_metrics.to_dict()
        p_row["feature_set"] = "None (Persistence)"
        p_row["num_features"] = 1
        records.append(p_row)
        print(f"[Persistence] Test MAE: {p_metrics.mae:.2f} mg/dL | RMSE: {p_metrics.rmse:.2f} | R2: {p_metrics.r2:.4f}")

        for fs_name, f_cols in feature_sets.items():
            print(f"  Training models with {fs_name} ({len(f_cols)} features)...")

            # Fit scaler on training subset ONLY (fill NAs for Ridge if any)
            # For Ridge: impute median for wearable columns if present
            train_sub = split.train_df.copy()
            val_sub = split.val_df.copy()
            test_sub = split.test_df.copy()

            for col in f_cols:
                if train_sub[col].isna().any():
                    med = train_sub[col].median()
                    train_sub[col] = train_sub[col].fillna(med)
                    val_sub[col] = val_sub[col].fillna(med)
                    test_sub[col] = test_sub[col].fillna(med)

            ds.fit_scaler(train_sub, feature_cols=f_cols)
            tr_scaled = ds.transform_df(train_sub, feature_cols=f_cols)
            va_scaled = ds.transform_df(val_sub, feature_cols=f_cols)
            te_scaled = ds.transform_df(test_sub, feature_cols=f_cols)

            tr_seq = builder.build_sequences(tr_scaled, feature_cols=f_cols)
            va_seq = builder.build_sequences(va_scaled, feature_cols=f_cols)
            te_seq = builder.build_sequences(te_scaled, feature_cols=f_cols)

            # Ground truth targets in original scale
            y_tr = train_base.y
            y_va = val_base.y
            y_te = test_base.y

            # 1. Ridge
            ridge = RidgeForecaster()
            best_alpha = ridge.tune_alpha(tr_seq.X, y_tr, va_seq.X, y_va, alpha_grid=[0.1, 1.0, 10.0, 100.0])
            r_preds = ridge.predict(te_seq.X)
            r_metrics = Evaluator.evaluate(
                y_true=y_te,
                y_pred=r_preds,
                model_name="Ridge",
                lookback_minutes=lookback_minutes,
                horizon_minutes=H,
            )
            r_row = r_metrics.to_dict()
            r_row["feature_set"] = fs_name
            r_row["num_features"] = len(f_cols)
            r_row["alpha"] = best_alpha
            records.append(r_row)

            # 2. XGBoost
            xgb_m = XGBoostForecaster(n_estimators=150, max_depth=5, learning_rate=0.05)
            xgb_m.fit(
                X_train=tr_seq.X,
                y_train=y_tr,
                X_val=va_seq.X,
                y_val=y_va,
                early_stopping_rounds=15,
            )
            x_preds = xgb_m.predict(te_seq.X)
            x_metrics = Evaluator.evaluate(
                y_true=y_te,
                y_pred=x_preds,
                model_name="XGBoost",
                lookback_minutes=lookback_minutes,
                horizon_minutes=H,
            )
            x_row = x_metrics.to_dict()
            x_row["feature_set"] = fs_name
            x_row["num_features"] = len(f_cols)
            records.append(x_row)

            print(
                f"    [Ridge   | {fs_name[:20]:20s}] Test MAE: {r_metrics.mae:5.2f} mg/dL | RMSE: {r_metrics.rmse:5.2f} | R2: {r_metrics.r2:6.4f} | ClinAcc: {r_metrics.clinical_accuracy_pct:.1f}%"
            )
            print(
                f"    [XGBoost | {fs_name[:20]:20s}] Test MAE: {x_metrics.mae:5.2f} mg/dL | RMSE: {x_metrics.rmse:5.2f} | R2: {x_metrics.r2:6.4f} | ClinAcc: {x_metrics.clinical_accuracy_pct:.1f}%"
            )

    df_out = pd.DataFrame(records)
    out_path = "results/feature_ablation_study.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nAblation study results saved to {out_path}!")

    print("\n" + "=" * 75)
    print("SUMMARY COMPARISON TABLE (TEST PARTITION)")
    print("=" * 75)
    cols_show = ["model_name", "feature_set", "horizon_minutes", "mae", "rmse", "r2", "clinical_accuracy_pct"]
    print(df_out[cols_show].to_string(index=False))


if __name__ == "__main__":
    run_ablation()

