"""Unit tests for src.monitoring.drift."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring.drift import (
    feature_drift_report,
    model_performance_over_time,
    population_stability_index,
    psi_status,
)


def test_psi_is_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, size=5000)
    psi = population_stability_index(pd.Series(data), pd.Series(data))
    assert psi < 1e-6


def test_psi_is_large_for_shifted_distribution() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, size=5000)
    shifted = rng.normal(3, 1, size=5000)
    psi = population_stability_index(pd.Series(reference), pd.Series(shifted))
    assert psi > 0.25


def test_psi_status_thresholds() -> None:
    assert psi_status(0.05, 0.10, 0.25) == "Stable"
    assert psi_status(0.15, 0.10, 0.25) == "Monitor"
    assert psi_status(0.30, 0.10, 0.25) == "Potential significant drift"


def test_feature_drift_report_flags_shifted_feature() -> None:
    rng = np.random.default_rng(1)
    reference_df = pd.DataFrame(
        {"stable_feat": rng.normal(0, 1, 3000), "shifted_feat": rng.normal(0, 1, 3000)}
    )
    current_df = pd.DataFrame(
        {"stable_feat": rng.normal(0, 1, 3000), "shifted_feat": rng.normal(4, 1, 3000)}
    )
    report = feature_drift_report(reference_df, current_df, ["stable_feat", "shifted_feat"])
    shifted_row = report[report["feature"] == "shifted_feat"].iloc[0]
    assert shifted_row["status"] == "Potential significant drift"


def test_model_performance_over_time_returns_expected_columns() -> None:
    rng = np.random.default_rng(2)
    n = 2000
    scored = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(rng.choice(["2024-01-15", "2024-02-15"], size=n)),
            "pd": rng.random(n),
            "target_default_90d": rng.binomial(1, 0.08, size=n),
        }
    )
    result = model_performance_over_time(scored)
    assert {"month", "roc_auc", "psi_vs_first_month", "psi_status"} <= set(result.columns)
    assert len(result) == 2
