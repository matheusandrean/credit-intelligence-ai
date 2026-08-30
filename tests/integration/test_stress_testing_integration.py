"""Integration tests for the full stress-test / what-if scoring path,
using real trained project artifacts. Skips if they are not present."""

from __future__ import annotations

import pandas as pd
import pytest

from src.risk.stress_testing import load_stress_scenarios, run_stress_test, run_what_if
from src.utils.config import get_settings


def _artifacts_available() -> bool:
    settings = get_settings()
    return (settings.models_path / "champion_calibrated.joblib").exists() and (
        settings.data_path / "processed" / "credit_features.parquet"
    ).exists()


@pytest.mark.skipif(not _artifacts_available(), reason="trained model artifacts not present")
def test_severe_scenario_increases_portfolio_pd_and_expected_loss() -> None:
    settings = get_settings()
    portfolio = pd.read_parquet(settings.data_path / "processed" / "credit_features.parquet")
    scenarios = load_stress_scenarios()

    baseline_result = run_stress_test(scenarios["baseline"], portfolio.sample(2000, random_state=1))
    severe_result = run_stress_test(scenarios["severe"], portfolio.sample(2000, random_state=1))

    assert severe_result["stressed"]["average_pd"] > baseline_result["baseline"]["average_pd"]
    assert severe_result["expected_loss_delta"] >= 0


@pytest.mark.skipif(not _artifacts_available(), reason="trained model artifacts not present")
def test_what_if_reports_simulation_flag_and_pd_movement() -> None:
    settings = get_settings()
    portfolio = pd.read_parquet(settings.data_path / "processed" / "credit_features.parquet")
    row = portfolio.iloc[[0]]

    result = run_what_if(row, income_shock_pct=-0.20, utilization_shock_pp=10.0)
    assert result["is_simulation"] is True
    assert result["simulated"]["pd"] >= result["baseline"]["pd"]
