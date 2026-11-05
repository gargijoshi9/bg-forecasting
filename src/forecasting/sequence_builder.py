"""
sequence_builder.py
===================
Leakage-safe sequence builder for continuous glucose time series.
Enforces:
1. Strict chronological ordering and timezone preservation.
2. Contiguous 5-minute spacing: sequences NEVER cross sensor gaps.
3. No synthetic data: missing intervals are NEVER interpolated.
4. Temporal anti-leakage: all input features in lookback are <= prediction time.
5. Preservation of coverage metadata and gap boundaries.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import pandas as pd


@dataclass
class SequenceDataset:
    """Container for generated time series sequences and evaluation metadata."""
    X: np.ndarray  # Shape: (N, lookback_steps, num_features)
    y: np.ndarray  # Shape: (N,) for single horizon, or (N, num_horizons)
    timestamps_pred: pd.Series
    timestamps_target: pd.Series
    metadata: pd.DataFrame
    feature_names: List[str]
    lookback_minutes: int
    horizon_minutes: int
    lookback_steps: int
    horizon_steps: int
    audit_stats: Dict[str, Any]

    def flatten_X(self) -> np.ndarray:
        """Returns X reshaped to 2D (N, lookback_steps * num_features) for tabular models."""
        return self.X.reshape(self.X.shape[0], -1)

    def __len__(self) -> int:
        return len(self.X)


class SequenceBuilder:
    """
    Constructs lookback and forecast horizon sequences from 5-minute CGM data.
    """

    def __init__(
        self,
        lookback_minutes: int = 60,
        horizon_minutes: int = 30,
        step_size_minutes: int = 5,
        target_col: str = "glucose",
        timestamp_col: str = "timestamp",
    ):
        """
        Args:
            lookback_minutes: Input window length in minutes (e.g., 60, 120, 360).
            horizon_minutes: Prediction horizon in minutes (e.g., 30, 60, 120).
            step_size_minutes: Expected sampling interval in minutes (5).
            target_col: Column name of prediction target.
            timestamp_col: Column name of timestamp.
        """
        if lookback_minutes <= 0 or horizon_minutes <= 0:
            raise ValueError("lookback_minutes and horizon_minutes must be positive.")
        if lookback_minutes % step_size_minutes != 0:
            raise ValueError(
                f"lookback_minutes ({lookback_minutes}) must be divisible by step_size ({step_size_minutes})."
            )
        if horizon_minutes % step_size_minutes != 0:
            raise ValueError(
                f"horizon_minutes ({horizon_minutes}) must be divisible by step_size ({step_size_minutes})."
            )

        self.lookback_minutes = lookback_minutes
        self.horizon_minutes = horizon_minutes
        self.step_size_minutes = step_size_minutes
        self.lookback_steps = lookback_minutes // step_size_minutes
        self.horizon_steps = horizon_minutes // step_size_minutes
        self.target_col = target_col
        self.timestamp_col = timestamp_col

    def _validate_input_df(self, df: pd.DataFrame, feature_cols: List[str]):
        """Validates timestamp ordering, duplicates, and column existence."""
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if self.timestamp_col not in df.columns:
            raise KeyError(f"Timestamp column '{self.timestamp_col}' not found.")
        if self.target_col not in df.columns:
            raise KeyError(f"Target column '{self.target_col}' not found.")
        for col in feature_cols:
            if col not in df.columns:
                raise KeyError(f"Feature column '{col}' not found in DataFrame.")

        # Check for duplicate timestamps
        duplicates = df[self.timestamp_col].duplicated()
        if duplicates.any():
            dup_count = duplicates.sum()
            raise ValueError(f"Found {dup_count} duplicate timestamps in input data.")

        # Check chronological sorting
        timestamps_dt = pd.to_datetime(df[self.timestamp_col])
        if not timestamps_dt.is_monotonic_increasing:
            raise ValueError("Input DataFrame timestamps are not monotonically increasing.")

    def find_contiguous_segments(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Identifies start and end indices [start, end] inclusive of contiguous blocks
        where consecutive rows are spaced by exactly step_size_minutes.
        """
        timestamps_dt = pd.to_datetime(df[self.timestamp_col])
        diffs = timestamps_dt.diff()
        expected_step = pd.Timedelta(minutes=self.step_size_minutes)

        # Non-contiguous transition points (gap boundaries)
        # Note: index 0 diff is NaT, which is the start of segment 0
        gap_mask = (diffs != expected_step) & (diffs.notna())
        gap_indices = df.index[gap_mask].tolist()

        segments = []
        start_idx = 0
        for gap_idx in gap_indices:
            end_idx = gap_idx - 1
            if end_idx >= start_idx:
                segments.append((start_idx, end_idx))
            start_idx = gap_idx
        # Append final segment
        segments.append((start_idx, len(df) - 1))
        return segments

    def build_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> SequenceDataset:
        """
        Extracts valid (X, y) sequence pairs from contiguous segments without gap crossing.

        Args:
            df: Continuous DataFrame of 5-minute features.
            feature_cols: Feature columns to include in X. If None, defaults to [target_col].

        Returns:
            SequenceDataset containing sequences, targets, timestamps, and audit metadata.
        """
        if feature_cols is None:
            feature_cols = [self.target_col]

        self._validate_input_df(df, feature_cols)

        total_rows = len(df)
        total_window_length = self.lookback_steps + self.horizon_steps
        theoretical_sliding_windows = max(0, total_rows - total_window_length + 1)

        segments = self.find_contiguous_segments(df)

        X_list = []
        y_list = []
        pred_times = []
        target_times = []
        meta_rows = []

        feature_matrix = df[feature_cols].to_numpy(dtype=np.float32)
        target_series = df[self.target_col].to_numpy(dtype=np.float32)
        timestamps = df[self.timestamp_col].to_numpy()
        reading_counts = (
            df["reading_count"].to_numpy(dtype=np.int32)
            if "reading_count" in df.columns
            else np.ones(total_rows, dtype=np.int32)
        )

        for seg_id, (seg_start, seg_end) in enumerate(segments):
            seg_len = seg_end - seg_start + 1
            if seg_len < total_window_length:
                # Segment is too short to produce even one valid (L + H) sequence
                continue

            num_valid_windows = seg_len - total_window_length + 1
            for offset in range(num_valid_windows):
                i = seg_start + offset
                pred_idx = i + self.lookback_steps - 1
                target_idx = pred_idx + self.horizon_steps

                # Slice lookback input features: [i : pred_idx + 1]
                x_seq = feature_matrix[i : pred_idx + 1]
                y_val = target_series[target_idx]

                t_pred = timestamps[pred_idx]
                t_target = timestamps[target_idx]

                # Coverage in lookback
                rc_window = reading_counts[i : pred_idx + 1]
                mean_rc = float(np.mean(rc_window))
                min_rc = int(np.min(rc_window))

                X_list.append(x_seq)
                y_list.append(y_val)
                pred_times.append(t_pred)
                target_times.append(t_target)

                meta_rows.append({
                    "segment_id": seg_id + 1,
                    "prediction_timestamp": t_pred,
                    "target_timestamp": t_target,
                    "lookback_steps": self.lookback_steps,
                    "horizon_steps": self.horizon_steps,
                    "reading_count_mean": round(mean_rc, 2),
                    "reading_count_min": min_rc,
                })

        valid_count = len(X_list)
        excluded_gaps = theoretical_sliding_windows - valid_count

        if valid_count > 0:
            X_arr = np.stack(X_list, axis=0)
            y_arr = np.array(y_list, dtype=np.float32)
        else:
            X_arr = np.empty((0, self.lookback_steps, len(feature_cols)), dtype=np.float32)
            y_arr = np.empty((0,), dtype=np.float32)

        meta_df = pd.DataFrame(meta_rows)

        audit_stats = {
            "total_rows": total_rows,
            "lookback_minutes": self.lookback_minutes,
            "horizon_minutes": self.horizon_minutes,
            "lookback_steps": self.lookback_steps,
            "horizon_steps": self.horizon_steps,
            "total_window_steps": total_window_length,
            "theoretical_sliding_windows": theoretical_sliding_windows,
            "valid_sequences_count": valid_count,
            "excluded_gap_windows": excluded_gaps,
            "contiguous_segments_count": len(segments),
        }

        return SequenceDataset(
            X=X_arr,
            y=y_arr,
            timestamps_pred=pd.Series(pred_times, name="prediction_timestamp"),
            timestamps_target=pd.Series(target_times, name="target_timestamp"),
            metadata=meta_df,
            feature_names=feature_cols,
            lookback_minutes=self.lookback_minutes,
            horizon_minutes=self.horizon_minutes,
            lookback_steps=self.lookback_steps,
            horizon_steps=self.horizon_steps,
            audit_stats=audit_stats,
        )

