"""
test_temporal_split.py
======================
Tests for temporal partitioning and leakage-safe scaling:
- Chronological ordering across splits (Train < Val < Test)
- Purge margin / gap isolation between partitions
- Training-only scaler fitting
"""

import unittest
import pandas as pd
import numpy as np
from src.forecasting.dataset import ForecastingDataset, TemporalSplit


class TestTemporalSplit(unittest.TestCase):

    def setUp(self):
        # Create a synthetic dataset matching the real schema
        dates = pd.date_range("2026-06-01 00:00", periods=500, freq="5min")
        self.df = pd.DataFrame({
            "timestamp": dates.strftime("%Y-%m-%dT%H:%M:%S+0530"),
            "glucose": np.random.normal(150, 30, len(dates)),
            "glucose_min": np.random.normal(148, 30, len(dates)),
            "glucose_max": np.random.normal(152, 30, len(dates)),
            "glucose_std": np.abs(np.random.normal(2, 1, len(dates))),
            "time_sin": np.sin(np.linspace(0, 2 * np.pi, len(dates))),
            "time_cos": np.cos(np.linspace(0, 2 * np.pi, len(dates))),
            "day_of_week": dates.dayofweek,
            "is_weekend": (dates.dayofweek >= 5).astype(int),
            "cycle_day": 10,
            "cycle_length_days": 28,
            "days_to_next_period": 18,
            "is_menstrual_flow": 0,
            "is_stressed": 0,
            "stress_intensity": 0,
            "reading_count": 5,
        })
        self.dataset = ForecastingDataset.__new__(ForecastingDataset)
        self.dataset.df = self.df
        self.dataset.feature_cols = [
            "glucose", "glucose_min", "glucose_max", "glucose_std",
            "time_sin", "time_cos", "day_of_week", "is_weekend",
            "cycle_day", "cycle_length_days", "days_to_next_period",
            "is_menstrual_flow", "is_stressed", "stress_intensity"
        ]
        self.dataset.scaler = None

    def test_chronological_ordering(self):
        """Verifies that Train is strictly before Val, which is strictly before Test."""
        split = self.dataset.create_fractional_split(train_frac=0.70, val_frac=0.15, test_frac=0.15, purge_steps=24)

        t_train_max = pd.to_datetime(split.train_df["timestamp"].iloc[-1])
        t_val_min = pd.to_datetime(split.val_df["timestamp"].iloc[0])
        t_val_max = pd.to_datetime(split.val_df["timestamp"].iloc[-1])
        t_test_min = pd.to_datetime(split.test_df["timestamp"].iloc[0])

        self.assertLess(t_train_max, t_val_min)
        self.assertLess(t_val_max, t_test_min)

    def test_purge_margin_isolation(self):
        """Verifies that purge margin completely separates partitions by at least purge_steps."""
        purge_steps = 36  # 36 * 5 = 180 mins
        split = self.dataset.create_fractional_split(train_frac=0.70, val_frac=0.15, test_frac=0.15, purge_steps=purge_steps)

        t_train_max = pd.to_datetime(split.train_df["timestamp"].iloc[-1])
        t_val_min = pd.to_datetime(split.val_df["timestamp"].iloc[0])
        diff_train_val = t_val_min - t_train_max

        # The difference must be strictly greater than purge_steps * 5min
        self.assertGreaterEqual(diff_train_val, pd.Timedelta(minutes=purge_steps * 5))

    def test_training_only_scaler_fitting(self):
        """Verifies that scaler parameters depend exclusively on the training partition."""
        split = self.dataset.create_fractional_split(train_frac=0.70, val_frac=0.15, test_frac=0.15)

        # Compute ground truth training mean for glucose
        expected_train_mean = split.train_df["glucose"].mean()

        # Fit scaler on train_df
        scaler = self.dataset.fit_scaler(split.train_df)
        fitted_mean = scaler.mean_[0]

        self.assertAlmostEqual(expected_train_mean, fitted_mean, places=4)

        # Modifying val_df or test_df must NOT alter fitted scaler parameters
        modified_val = split.val_df.copy()
        modified_val["glucose"] = 9999.0
        # Transform val_df
        scaled_val = self.dataset.transform_df(modified_val)
        self.assertAlmostEqual(self.dataset.scaler.mean_[0], expected_train_mean, places=4)


if __name__ == "__main__":
    unittest.main()

