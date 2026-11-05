"""
ridge_model.py
==============
Ridge regression model for multi-horizon blood glucose forecasting.
Flattens lookback sequences into regularized linear feature representations.
"""

from typing import List, Optional, Dict, Any, Union
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


class RidgeForecaster:
    """
    Ridge regression model for glucose forecasting.
    Accepts 3D sequence arrays (N, lookback_steps, num_features) or 2D flattened arrays (N, num_features).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.model = Ridge(
            alpha=alpha,
            fit_intercept=fit_intercept,
            random_state=random_state,
        )
        self.is_fitted = False
        self.best_alpha: Optional[float] = None
        self.feature_names: Optional[List[str]] = None

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        """Flattens 3D sequences (N, L, D) into 2D (N, L*D)."""
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        elif X.ndim == 2:
            return X
        else:
            raise ValueError(f"Expected 2D or 3D array for X, got shape {X.shape}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ):
        """Fits Ridge model on training data."""
        X_flat = self._prepare_X(X)
        if len(X_flat) != len(y):
            raise ValueError(f"Mismatched lengths: X has {len(X_flat)}, y has {len(y)}")

        self.model.fit(X_flat, y)
        self.is_fitted = True
        self.best_alpha = self.alpha
        self.feature_names = feature_names
        return self

    def tune_alpha(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        alpha_grid: Optional[List[float]] = None,
    ) -> float:
        """
        Tunes regularization parameter alpha on validation data (minimizing MAE).
        """
        if alpha_grid is None:
            alpha_grid = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

        X_tr = self._prepare_X(X_train)
        X_v = self._prepare_X(X_val)

        best_score = float("inf")
        best_a = self.alpha

        for a in alpha_grid:
            m = Ridge(alpha=a, fit_intercept=self.fit_intercept, random_state=self.random_state)
            m.fit(X_tr, y_train)
            preds = m.predict(X_v)
            mae = mean_absolute_error(y_val, preds)
            if mae < best_score:
                best_score = mae
                best_a = a

        self.alpha = best_a
        self.best_alpha = best_a
        self.model = Ridge(alpha=best_a, fit_intercept=self.fit_intercept, random_state=self.random_state)
        self.model.fit(X_tr, y_train)
        self.is_fitted = True
        return best_a

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates predictions for X."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        X_flat = self._prepare_X(X)
        preds = self.model.predict(X_flat)
        return preds.astype(np.float32)

    @property
    def coefficients(self) -> np.ndarray:
        """Returns model weights."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")
        return self.model.coef_

    @property
    def intercept(self) -> float:
        """Returns intercept value."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")
        return float(self.model.intercept_)

