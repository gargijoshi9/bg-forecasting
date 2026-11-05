"""
evaluation.py
=============
Clinical and statistical evaluation metrics for blood glucose forecasting.
Computes:
1. MAE (Mean Absolute Error, mg/dL)
2. RMSE (Root Mean Squared Error, mg/dL)
3. R² (Coefficient of Determination, supplementary)
4. MAPE (Mean Absolute Percentage Error, %)
5. Clinical Consensus Accuracy: % of predictions within +/-15 mg/dL (<100 mg/dL) or +/-20% (>=100 mg/dL).
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class ForecastMetrics:
    model_name: str
    lookback_minutes: int
    horizon_minutes: int
    sample_count: int
    mae: float
    rmse: float
    r2: float
    mape: float
    clinical_accuracy_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "lookback_minutes": self.lookback_minutes,
            "horizon_minutes": self.horizon_minutes,
            "sample_count": self.sample_count,
            "mae": round(self.mae, 2),
            "rmse": round(self.rmse, 2),
            "r2": round(self.r2, 4),
            "mape": round(self.mape, 2),
            "clinical_accuracy_pct": round(self.clinical_accuracy_pct, 2),
        }


class Evaluator:
    """Evaluates forecast predictions against ground truth targets."""

    @staticmethod
    def compute_clinical_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates % of predictions meeting clinical CGM consensus standards:
        - Within +/- 15 mg/dL when reference glucose < 100 mg/dL
        - Within +/- 20% when reference glucose >= 100 mg/dL
        """
        diff = np.abs(y_true - y_pred)
        low_ref_mask = y_true < 100.0
        high_ref_mask = ~low_ref_mask

        compliant = np.zeros(len(y_true), dtype=bool)
        if low_ref_mask.any():
            compliant[low_ref_mask] = diff[low_ref_mask] <= 15.0
        if high_ref_mask.any():
            pct_diff = (diff[high_ref_mask] / y_true[high_ref_mask]) * 100.0
            compliant[high_ref_mask] = pct_diff <= 20.0

        return float(np.mean(compliant) * 100.0)

    @classmethod
    def evaluate(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        lookback_minutes: int,
        horizon_minutes: int,
    ) -> ForecastMetrics:
        """Computes statistical and clinical forecast metrics."""
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}")
        if len(y_true) == 0:
            raise ValueError("Target array is empty.")

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
        clin_acc = cls.compute_clinical_accuracy(y_true, y_pred)

        return ForecastMetrics(
            model_name=model_name,
            lookback_minutes=lookback_minutes,
            horizon_minutes=horizon_minutes,
            sample_count=len(y_true),
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            clinical_accuracy_pct=clin_acc,
        )

    @staticmethod
    def metrics_to_dataframe(metrics_list: List[ForecastMetrics]) -> pd.DataFrame:
        """Converts a list of ForecastMetrics to a structured DataFrame."""
        return pd.DataFrame([m.to_dict() for m in metrics_list])

