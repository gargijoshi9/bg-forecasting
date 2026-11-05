"""
test_sequence_builder.py
========================
Tests for SequenceBuilder:
- Gap crossing prevention
- Missing readings handling
- Duplicate timestamp rejection
- Out-of-order timestamp rejection
- Multi-lookback and multi-horizon validation
- Metadata and coverage preservation
"""

import unittest
import pandas as pd
import numpy as np
from src.forecasting.sequence_builder import SequenceBuilder


class TestSequenceBuilder(unittest.TestCase):

    def setUp(self):
        # Create a synthetic contiguous 5-minute dataset with known gap
        # Block 1: 20 points (0 to 95 mins)
        dates1 = pd.date_range("2026-06-01 08:00", periods=20, freq="5min")
        # Gap of 4 hours
        # Block 2: 20 points (starting 4 hours later)
        dates2 = pd.date_range("2026-06-01 13:40", periods=20, freq="5min")

        timestamps = list(dates1.strftime("%Y-%m-%dT%H:%M:%S+0530")) + list(dates2.strftime("%Y-%m-%dT%H:%M:%S+0530"))
        glucose = np.linspace(100, 200, len(timestamps))

        self.df_with_gap = pd.DataFrame({
            "timestamp": timestamps,
            "glucose": glucose,
            "reading_count": 5,
        })

    def test_no_gap_crossing(self):
        """Verifies that no generated sequence crosses the 4-hour gap between Block 1 and Block 2."""
        builder = SequenceBuilder(lookback_minutes=30, horizon_minutes=15, step_size_minutes=5)
        # L = 6 steps, H = 3 steps, total window = 9 steps.
        # Block 1 has 20 points -> 20 - 9 + 1 = 12 valid sequences.
        # Block 2 has 20 points -> 20 - 9 + 1 = 12 valid sequences.
        # Total valid sequences = 24.
        # Total naive sliding windows would have been 40 - 9 + 1 = 32.
        # 8 sequences must be excluded because of the gap!
        seq_ds = builder.build_sequences(self.df_with_gap, feature_cols=["glucose"])

        self.assertEqual(len(seq_ds), 24)
        self.assertEqual(seq_ds.audit_stats["excluded_gap_windows"], 8)
        self.assertEqual(seq_ds.audit_stats["valid_sequences_count"], 24)

        # Verify that for every sequence, prediction_time and target_time are within the same block!
        for i in range(len(seq_ds)):
            t_pred = pd.to_datetime(seq_ds.timestamps_pred.iloc[i])
            t_target = pd.to_datetime(seq_ds.timestamps_target.iloc[i])
            self.assertEqual(t_target - t_pred, pd.Timedelta(minutes=15))

    def test_multiple_lookbacks_and_horizons(self):
        """Tests that (60, 120, 360) lookbacks and (30, 60, 120) horizons compute correctly."""
        # Create a large contiguous block of 200 points (1000 mins)
        dates = pd.date_range("2026-06-01 00:00", periods=200, freq="5min")
        df_large = pd.DataFrame({
            "timestamp": dates.strftime("%Y-%m-%dT%H:%M:%S+0530"),
            "glucose": np.random.uniform(100, 200, 200),
            "reading_count": 5,
        })

        configs = [
            (60, 30),   # L=12, H=6  -> W=18 -> expected = 200 - 18 + 1 = 183
            (60, 60),   # L=12, H=12 -> W=24 -> expected = 200 - 24 + 1 = 177
            (60, 120),  # L=12, H=24 -> W=36 -> expected = 200 - 36 + 1 = 165
            (120, 30),  # L=24, H=6  -> W=30 -> expected = 200 - 30 + 1 = 171
            (120, 60),  # L=24, H=12 -> W=36 -> expected = 200 - 36 + 1 = 165
            (120, 120), # L=24, H=24 -> W=48 -> expected = 200 - 48 + 1 = 153
            (360, 30),  # L=72, H=6  -> W=78 -> expected = 200 - 78 + 1 = 123
            (360, 60),  # L=72, H=12 -> W=84 -> expected = 200 - 84 + 1 = 117
            (360, 120), # L=72, H=24 -> W=96 -> expected = 200 - 96 + 1 = 105
        ]

        for L_min, H_min in configs:
            builder = SequenceBuilder(lookback_minutes=L_min, horizon_minutes=H_min)
            seq_ds = builder.build_sequences(df_large, feature_cols=["glucose"])
            expected_n = 200 - (L_min // 5 + H_min // 5) + 1
            self.assertEqual(len(seq_ds), expected_n)
            self.assertEqual(seq_ds.X.shape, (expected_n, L_min // 5, 1))
            self.assertEqual(seq_ds.y.shape, (expected_n,))

    def test_duplicate_timestamps_rejected(self):
        """Verifies that duplicate timestamps raise ValueError."""
        df_dup = self.df_with_gap.copy()
        df_dup.loc[5, "timestamp"] = df_dup.loc[4, "timestamp"]
        builder = SequenceBuilder(lookback_minutes=30, horizon_minutes=15)
        with self.assertRaises(ValueError):
            builder.build_sequences(df_dup)

    def test_out_of_order_timestamps_rejected(self):
        """Verifies that non-monotonic timestamps raise ValueError."""
        df_shuffled = self.df_with_gap.sample(frac=1.0, random_state=42).reset_index(drop=True)
        builder = SequenceBuilder(lookback_minutes=30, horizon_minutes=15)
        with self.assertRaises(ValueError):
            builder.build_sequences(df_shuffled)

    def test_missing_column_rejected(self):
        """Verifies that missing requested feature column raises KeyError."""
        builder = SequenceBuilder(lookback_minutes=30, horizon_minutes=15)
        with self.assertRaises(KeyError):
            builder.build_sequences(self.df_with_gap, feature_cols=["nonexistent_col"])

    def test_metadata_coverage_preservation(self):
        """Verifies reading_count coverage metadata is preserved."""
        builder = SequenceBuilder(lookback_minutes=30, horizon_minutes=15)
        seq_ds = builder.build_sequences(self.df_with_gap, feature_cols=["glucose"])
        self.assertIn("reading_count_mean", seq_ds.metadata.columns)
        self.assertIn("reading_count_min", seq_ds.metadata.columns)
        self.assertEqual(seq_ds.metadata["reading_count_mean"].iloc[0], 5.0)


if __name__ == "__main__":
    unittest.main()

