"""
Forecasting module for continuous glucose monitoring (CGM).
Provides leakage-safe sequence generation, temporal evaluation, baselines,
models, and experiment tracking.
"""

from .sequence_builder import SequenceBuilder, SequenceDataset
from .dataset import ForecastingDataset, TemporalSplit
from .baseline import PersistenceBaseline, MeanBaseline
from .ridge_model import RidgeForecaster
from .xgboost_model import XGBoostForecaster
from .evaluation import Evaluator, ForecastMetrics
from .experiment_tracker import ExperimentTracker

__all__ = [
    "SequenceBuilder",
    "SequenceDataset",
    "ForecastingDataset",
    "TemporalSplit",
    "PersistenceBaseline",
    "MeanBaseline",
    "RidgeForecaster",
    "XGBoostForecaster",
    "Evaluator",
    "ForecastMetrics",
    "ExperimentTracker",
]

