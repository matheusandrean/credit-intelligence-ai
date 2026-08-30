"""Shared, lazily-loaded resources for the LLM tools.

Loads the scored portfolio, feature dataset, model metadata, SHAP explainer
and an in-memory DuckDB connection exactly once per process and reuses them
across tool calls, instead of re-reading parquet files on every question.
"""

from __future__ import annotations

import json
import os
from functools import cached_property

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import duckdb
import joblib
import pandas as pd

from src.models.explainability import ShapExplainer
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ToolContext:
    """Holds every resource the agent tools need, loaded on first access."""

    def __init__(self) -> None:
        self._settings = get_settings()

    @cached_property
    def model_metadata(self) -> dict:
        path = self._settings.models_path / "model_metadata.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @cached_property
    def features(self) -> pd.DataFrame:
        path = self._settings.data_path / "processed" / "credit_features.parquet"
        return pd.read_parquet(path)

    @cached_property
    def scored_portfolio(self) -> pd.DataFrame:
        path = self._settings.data_path / "processed" / "scored_portfolio.parquet"
        return pd.read_parquet(path)

    @cached_property
    def monthly_performance(self) -> pd.DataFrame:
        path = self._settings.data_path / "raw" / "monthly_performance.parquet"
        return pd.read_parquet(path)

    @cached_property
    def customers_raw(self) -> pd.DataFrame:
        path = self._settings.data_path / "raw" / "customer_portfolio.parquet"
        return pd.read_parquet(path)

    @cached_property
    def champion_model(self):
        name = self.model_metadata["champion_model"]
        return joblib.load(self._settings.models_path / f"{name}.joblib")

    @cached_property
    def calibrated_model(self):
        path = self._settings.models_path / self.model_metadata.get(
            "calibrated_model_path", "champion_calibrated.joblib"
        )
        return joblib.load(path)

    @cached_property
    def shap_explainer(self) -> ShapExplainer:
        from src.models.train import load_dataset, split_xy

        champion_name = self.model_metadata["champion_model"]
        categories = self.model_metadata["categories"]
        df = load_dataset()
        x_train, _ = split_xy(df, "train", categories)
        background = x_train.sample(min(300, len(x_train)), random_state=42)
        return ShapExplainer(champion_name, self.champion_model, background)

    @cached_property
    def duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(database=":memory:")
        portfolio_view = self.scored_portfolio.merge(
            self.features.drop(columns=["target_default_90d"], errors="ignore"),
            on="customer_id",
            how="left",
            suffixes=("", "_feat"),
        )
        con.register("portfolio", portfolio_view)
        return con

    def get_customer_row(self, customer_id: str) -> pd.DataFrame | None:
        row = self.features[self.features["customer_id"] == customer_id]
        if row.empty:
            return None
        return row


_default_context: ToolContext | None = None


def get_tool_context() -> ToolContext:
    global _default_context
    if _default_context is None:
        _default_context = ToolContext()
    return _default_context


def reset_tool_context() -> None:
    """Used by tests to force fresh resource loading."""
    global _default_context
    _default_context = None
