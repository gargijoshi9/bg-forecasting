"""
baseline.py
===========
Baseline forecasting models for blood glucose:
1. PersistenceBaseline: Forecast equals the latest observed glucose value in the lookback window.
2. MeanBaseline: Forecast equals historical training target mean.
"""

from typing import Optional, Union
import numpy as np


class PersistenceBaseline:
    """
    Persistence baseline for blood glucose forecasting.
    Predicts that glucose at horizon t + H equals the latest observed glucose at time t:
        y_hat_{t+H} = y_t
    """

    def __init__(self, glucose_idx: int = 0):
        """
        Args:
            glucose_idx: Index of glucose feature in the feature dimension.
        """
        self.glucose_idx = glucose_idx
        self.is_fitted = True

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """No-op for persistence baseline."""
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates persistence forecast from input sequence X.

        Args:
            X: Array of shape (N, lookback_steps, num_features) or (N, lookback_steps).

        Returns:
            1D array of shape (N,) containing latest glucose value for each sequence.
        """
        if X.ndim == 3:
            # (N, L, D) -> latest step is index -1
            return X[:, -1, self.glucose_idx].astype(np.float32)
        elif X.ndim == 2:
            # (N, L) -> univariate glucose sequence
            return X[:, -1].astype(np.float32)
        else:
            raise ValueError(f"Expected 2D or 3D array for X, got shape {X.shape}")


class MeanBaseline:
    """
    Predicts constant historical training mean:
        y_hat_{t+H} = mean(y_train)
    """

    def __init__(self):
        self.mean_val: Optional[float] = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits historical mean on training target y."""
        if y is None or len(y) == 0:
            raise ValueError("Target array y must not be empty.")
        self.mean_val = float(np.mean(y))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts constant historical mean for all samples."""
        if not self.is_fitted or self.mean_val is None:
            raise RuntimeError("Model has not been fitted.")
        return np.full((X.shape[0],), self.mean_val, dtype=np.float32)

