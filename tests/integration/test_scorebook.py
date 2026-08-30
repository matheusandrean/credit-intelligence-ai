"""Integration test for the end-to-end scored portfolio artifact.

Uses the real, already-trained project artifacts (models/, data/processed/)
produced by `make data && make features && make train && make validate`.
Skips gracefully if those artifacts are not present (e.g. a fresh clone
before the pipeline has been run).
"""

from __future__ import annotations

import pytest

from src.risk.scorebook import build_scored_portfolio
from src.utils.config import get_settings


def _artifacts_available() -> bool:
    settings = get_settings()
    model_meta = settings.models_path / "model_metadata.json"
    features = settings.data_path / "processed" / "credit_features.parquet"
    return model_meta.exists() and features.exists()


@pytest.mark.skipif(not _artifacts_available(), reason="trained model artifacts not present")
def test_scored_portfolio_has_expected_columns_and_ranges() -> None:
    scored = build_scored_portfolio()
    expected_cols = {"customer_id", "pd", "score", "risk_band", "lgd", "ead", "expected_loss"}
    assert expected_cols <= set(scored.columns)
    assert scored["pd"].between(0, 1).all()
    assert scored["score"].between(300, 900).all()
    assert set(scored["risk_band"].unique()) <= {"A", "B", "C", "D", "E"}
    assert (scored["expected_loss"] >= 0).all()


@pytest.mark.skipif(not _artifacts_available(), reason="trained model artifacts not present")
def test_default_rate_increases_from_band_a_to_band_e() -> None:
    scored = build_scored_portfolio()
    rates = scored.groupby("risk_band")["target_default_90d"].mean()
    ordered = [rates[b] for b in ["A", "B", "C", "D", "E"] if b in rates.index]
    assert ordered == sorted(ordered)
