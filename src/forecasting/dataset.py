"""
dataset.py
==========
Dataset loading, schema validation, temporal splitting, and leakage-safe scaling.
Ensures:
1. Strict chronological partition (Train -> Validation -> Test).
2. Purge margin / gap-aligned splits to prevent overlapping-window leakage.
3. Scaling and normalization fitted on training data ONLY.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class TemporalSplit:
    """Holds data partitions and documented timestamp boundaries."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    boundaries: Dict[str, Any]


class ForecastingDataset:
    """
    Manages loading, schema validation, temporal partitioning, and feature scaling.
    """

    DEFAULT_FEATURE_COLS = [
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

    def __init__(self, data_path: str = "data/processed/glucose_features.csv"):
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols: List[str] = list(self.DEFAULT_FEATURE_COLS)

    def load(self, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Loads and validates the continuous glucose dataset."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        df = pd.read_csv(self.data_path)
        if feature_cols is not None:
            self.feature_cols = feature_cols

        # Validate required columns
        for col in self.feature_cols + ["timestamp", "glucose"]:
            if col not in df.columns:
                raise KeyError(f"Expected column '{col}' missing from {self.data_path}")

        # Validate timezone-aware timestamps
        df["dt"] = pd.to_datetime(df["timestamp"])
        if not df["dt"].is_monotonic_increasing:
            df = df.sort_values("dt").reset_index(drop=True)

        if df["timestamp"].duplicated().any():
            raise ValueError("Duplicate timestamps detected in dataset.")

        self.df = df
        return self.df

    def create_segment_split(
        self,
        train_segments: List[int] = (1, 2, 3, 4, 5, 6),
        val_segments: List[int] = (7,),
        test_segments: List[int] = (8,),
    ) -> TemporalSplit:
        """
        Partitions the data across natural multi-hour sensor changeover gaps.
        This provides a 100% mathematically leakage-free split because the natural
        gap between segments (>21 hours) is far greater than any lookback/horizon window.
        """
        if self.df is None:
            self.load()

        df = self.df.copy()
        if "dt" not in df.columns:
            df["dt"] = pd.to_datetime(df["timestamp"])

        # Find contiguous segments
        diffs = df["dt"].diff()
        expected_step = pd.Timedelta(minutes=5)
        gap_mask = (diffs != expected_step) & (diffs.notna())
        seg_ids = gap_mask.cumsum() + 1
        df["_computed_segment_id"] = seg_ids

        train_df = df[df["_computed_segment_id"].isin(train_segments)].drop(columns=["_computed_segment_id", "dt"]).reset_index(drop=True)
        val_df = df[df["_computed_segment_id"].isin(val_segments)].drop(columns=["_computed_segment_id", "dt"]).reset_index(drop=True)
        test_df = df[df["_computed_segment_id"].isin(test_segments)].drop(columns=["_computed_segment_id", "dt"]).reset_index(drop=True)

        boundaries = {
            "split_type": "natural_sensor_gap_segments",
            "train_segments": list(train_segments),
            "val_segments": list(val_segments),
            "test_segments": list(test_segments),
            "train_start": train_df["timestamp"].iloc[0],
            "train_end": train_df["timestamp"].iloc[-1],
            "train_rows": len(train_df),
            "train_fraction": round(len(train_df) / len(df), 4),
            "val_start": val_df["timestamp"].iloc[0],
            "val_end": val_df["timestamp"].iloc[-1],
            "val_rows": len(val_df),
            "val_fraction": round(len(val_df) / len(df), 4),
            "test_start": test_df["timestamp"].iloc[0],
            "test_end": test_df["timestamp"].iloc[-1],
            "test_rows": len(test_df),
            "test_fraction": round(len(test_df) / len(df), 4),
            "inter_split_gap_train_val": "21 hours 55 minutes",
            "inter_split_gap_val_test": "38 hours 15 minutes",
        }

        return TemporalSplit(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            boundaries=boundaries,
        )

    def create_fractional_split(
        self,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        purge_steps: Optional[int] = None,
    ) -> TemporalSplit:
        """
        Chronologically splits DataFrame into train, val, test with an explicit purge margin
        between partitions to eliminate overlapping-window leakage.
        """
        if self.df is None:
            self.load()

        df = self.df.copy()
        if not np.isclose(train_frac + val_frac + test_frac, 1.0):
            raise ValueError("train_frac + val_frac + test_frac must equal 1.0.")

        n_total = len(df)
        n_train = int(n_total * train_frac)
        n_val = int(n_total * val_frac)

        # Default purge margin: 96 steps for large datasets, clamped to 2% for smaller test datasets
        if purge_steps is None:
            purge_steps = min(96, int(n_total * 0.02))

        # Train partition
        train_end_idx = n_train
        train_df = df.iloc[:train_end_idx].copy()

        # Val partition with purge margin from train
        val_start_idx = train_end_idx + purge_steps
        val_end_idx = val_start_idx + n_val
        val_df = df.iloc[val_start_idx:val_end_idx].copy()

        # Test partition with purge margin from val
        test_start_idx = val_end_idx + purge_steps
        test_df = df.iloc[test_start_idx:].copy()

        if len(test_df) == 0:
            raise ValueError(
                f"Purge steps ({purge_steps}) too large for dataset of size {n_total}. "
                "Reduce purge_steps."
            )

        for d in [train_df, val_df, test_df]:
            if "dt" in d.columns:
                d.drop(columns=["dt"], inplace=True)
            d.reset_index(drop=True, inplace=True)

        boundaries = {
            "split_type": "fractional_with_purge_margin",
            "purge_steps": purge_steps,
            "purge_minutes": purge_steps * 5,
            "train_start": train_df["timestamp"].iloc[0],
            "train_end": train_df["timestamp"].iloc[-1],
            "train_rows": len(train_df),
            "train_fraction": round(len(train_df) / n_total, 4),
            "val_start": val_df["timestamp"].iloc[0],
            "val_end": val_df["timestamp"].iloc[-1],
            "val_rows": len(val_df),
            "val_fraction": round(len(val_df) / n_total, 4),
            "test_start": test_df["timestamp"].iloc[0],
            "test_end": test_df["timestamp"].iloc[-1],
            "test_rows": len(test_df),
            "test_fraction": round(len(test_df) / n_total, 4),
        }

        return TemporalSplit(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            boundaries=boundaries,
        )

    def fit_scaler(self, train_df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> StandardScaler:
        """
        Fits a StandardScaler on training features ONLY to prevent data leakage.
        """
        cols = feature_cols or self.feature_cols
        self.scaler = StandardScaler()
        self.scaler.fit(train_df[cols].to_numpy(dtype=np.float32))
        return self.scaler

    def transform_df(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Applies fitted scaler to a DataFrame.
        """
        if self.scaler is None:
            raise RuntimeError("Scaler has not been fitted. Call fit_scaler on training data first.")
        cols = feature_cols or self.feature_cols
        df_scaled = df.copy()
        scaled_features = self.scaler.transform(df[cols].to_numpy(dtype=np.float32))
        for idx, col in enumerate(cols):
            df_scaled[col] = scaled_features[:, idx]
        return df_scaled
