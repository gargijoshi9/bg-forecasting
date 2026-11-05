"""
experiment_tracker.py
=====================
Logs and persists experimental runs, hyperparameters, and evaluation metrics.
Saves outputs to results/forecasting_experiments.csv and markdown summaries.
"""

from typing import List, Dict, Any, Optional
import os
import json
import pandas as pd
from .evaluation import ForecastMetrics


class ExperimentTracker:
    """Tracks, formats, and persists forecast modeling experiments."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.records: List[Dict[str, Any]] = []

    def log_run(
        self,
        metrics: ForecastMetrics,
        split_name: str,
        hyperparams: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ):
        """Logs a single experimental evaluation run."""
        record = metrics.to_dict()
        record["split"] = split_name
        if hyperparams:
            for k, v in hyperparams.items():
                record[f"hp_{k}"] = v
        if extra_meta:
            for k, v in extra_meta.items():
                record[f"meta_{k}"] = v
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Returns all logged experiment records as a DataFrame."""
        return pd.DataFrame(self.records)

    def save(self, filename: str = "forecasting_experiments.csv") -> str:
        """Saves experiment results to CSV."""
        df = self.to_dataframe()
        out_path = os.path.join(self.output_dir, filename)
        df.to_csv(out_path, index=False)
        return out_path

    def generate_markdown_summary(self, split: str = "test") -> str:
        """Generates a GitHub-flavored markdown table for a given split without external dependencies."""
        df = self.to_dataframe()
        if df.empty:
            return "No experiments recorded."

        if "split" in df.columns:
            df = df[df["split"] == split]

        cols = [
            "model_name",
            "lookback_minutes",
            "horizon_minutes",
            "mae",
            "rmse",
            "r2",
            "clinical_accuracy_pct",
        ]
        available_cols = [c for c in cols if c in df.columns]
        sub_df = df[available_cols].copy()

        # Build Markdown table
        header = "| " + " | ".join(available_cols) + " |"
        sep = "| " + " | ".join(["---"] * len(available_cols)) + " |"
        rows = []
        for _, r in sub_df.iterrows():
            row_str = "| " + " | ".join(str(r[c]) for c in available_cols) + " |"
            rows.append(row_str)

        return "\n".join([header, sep] + rows)
