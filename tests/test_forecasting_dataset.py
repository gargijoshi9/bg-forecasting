"""
test_forecasting_dataset.py
===========================
Tests for baseline models, Ridge regression, and evaluation metrics:
- Persistence baseline exact matching
- Mean baseline exact matching
- Ridge regression training, tuning, and prediction
- Evaluator metric calculations (MAE, RMSE, R2, clinical accuracy)
"""

import unittest
import numpy as np
import pandas as pd
from src.forecasting.baseline import PersistenceBaseline, MeanBaseline
from src.forecasting.ridge_model import RidgeForecaster
from src.forecasting.evaluation import Evaluator, ForecastMetrics


class TestForecastingModelsAndEvaluation(unittest.TestCase):

    def setUp(self):
        # 100 sequences, lookback 12 (60 min), 3 features (glucose at index 0)
        np.random.seed(42)
        self.N = 100
        self.L = 12
        self.D = 3

        self.X = np.random.uniform(70, 250, (self.N, self.L, self.D)).astype(np.float32)
        # Create ground truth target that is slightly shifted from latest glucose
        latest_glucose = self.X[:, -1, 0]
        self.y = (latest_glucose + np.random.normal(0, 5, self.N)).astype(np.float32)

    def test_persistence_baseline_correctness(self):
        """Verifies persistence baseline equals the latest observed glucose value."""
        model = PersistenceBaseline(glucose_idx=0)
        preds = model.predict(self.X)

        expected = self.X[:, -1, 0]
        np.testing.assert_array_almost_equal(preds, expected)

    def test_mean_baseline_correctness(self):
        """Verifies mean baseline predicts constant historical mean."""
        model = MeanBaseline()
        model.fit(self.X, self.y)
        preds = model.predict(self.X)

        expected_mean = float(np.mean(self.y))
        self.assertTrue(np.allclose(preds, expected_mean))

    def test_ridge_forecaster_fit_and_predict(self):
        """Verifies RidgeForecaster trains and predicts valid output dimensions."""
        model = RidgeForecaster(alpha=1.0)
        model.fit(self.X, self.y)

        preds = model.predict(self.X)
        self.assertEqual(preds.shape, (self.N,))
        self.assertEqual(len(model.coefficients), self.L * self.D)

    def test_ridge_alpha_tuning(self):
        """Verifies alpha tuning executes and selects an alpha from grid."""
        X_train, y_train = self.X[:70], self.y[:70]
        X_val, y_val = self.X[70:], self.y[70:]

        model = RidgeForecaster()
        best_alpha = model.tune_alpha(X_train, y_train, X_val, y_val, alpha_grid=[0.1, 1.0, 10.0])
        self.assertIn(best_alpha, [0.1, 1.0, 10.0])
        self.assertEqual(model.best_alpha, best_alpha)

    def test_evaluator_metrics(self):
        """Verifies that Evaluator computes exact known mathematical metrics."""
        y_true = np.array([100.0, 200.0, 150.0])
        y_pred = np.array([110.0, 190.0, 150.0])
        # absolute errors: 10, 10, 0 -> MAE = 20/3 = 6.6667
        # squared errors: 100, 100, 0 -> MSE = 200/3 = 66.6667 -> RMSE = 8.165
        metrics = Evaluator.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            model_name="TestModel",
            lookback_minutes=60,
            horizon_minutes=30,
        )

        self.assertAlmostEqual(metrics.mae, 20.0 / 3.0, places=3)
        self.assertAlmostEqual(metrics.rmse, np.sqrt(200.0 / 3.0), places=3)
        self.assertEqual(metrics.sample_count, 3)
        # All errors are <= 10%, so 100% compliant with clinical accuracy
        self.assertEqual(metrics.clinical_accuracy_pct, 100.0)


if __name__ == "__main__":
    unittest.main()

