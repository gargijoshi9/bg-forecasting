"""
xgboost_model.py
================
XGBoost regressor for multi-horizon blood glucose forecasting.
Supports:
1. Flattens 3D sequences into tabular representations.
2. Native missing value handling (vital for intermittent wearable HR/HRV).
3. Early stopping on validation set.
4. Feature importance extraction (gain, weight).
"""

from typing import List, Optional, Dict, Any, Union
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error


class XGBoostForecaster:
    """
    Gradient boosted decision trees for glucose forecasting.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs,
            objective="reg:squarederror",
            tree_method="hist",
        )
        self.is_fitted = False
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
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        early_stopping_rounds: Optional[int] = 15,
    ):
        """Fits XGBoost model with optional validation-based early stopping."""
        X_tr = self._prepare_X(X_train)
        self.feature_names = feature_names

        if X_val is not None and y_val is not None and early_stopping_rounds is not None:
            X_v = self._prepare_X(X_val)
            self.model.set_params(early_stopping_rounds=early_stopping_rounds)
            self.model.fit(
                X_tr,
                y_train,
                eval_set=[(X_v, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_tr, y_train)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates predictions for X."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        X_flat = self._prepare_X(X)
        preds = self.model.predict(X_flat)
        return preds.astype(np.float32)

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns feature importances mapped to feature names if available."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")
        scores = self.model.feature_importances_
        if self.feature_names is not None and len(self.feature_names) == len(scores):
            return dict(zip(self.feature_names, [float(s) for s in scores]))
        return {f"f{i}": float(s) for i, s in enumerate(scores)}

