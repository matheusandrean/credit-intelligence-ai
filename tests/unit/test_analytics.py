"""Unit tests for src.analytics (portfolio, vintage, roll_rate)."""

from __future__ import annotations

import pandas as pd

from src.analytics.portfolio import compare_segments, monthly_kpi_trend, portfolio_summary
from src.analytics.roll_rate import build_roll_rate_matrix, cure_and_migration_rates
from src.analytics.vintage import build_vintage_curves, cohort_summary, filter_material_cohorts
from src.data.schema import DELINQUENCY_BUCKETS


def _sample_scored_and_features():
    scored = pd.DataFrame(
        {
            "customer_id": [f"CUST_{i:06d}" for i in range(6)],
            "observation_date": pd.to_datetime(["2024-01-31"] * 3 + ["2024-02-29"] * 3),
            "split": "train",
            "pd": [0.01, 0.05, 0.30, 0.02, 0.10, 0.40],
            "score": [850, 750, 550, 830, 700, 500],
            "risk_band": ["A", "B", "D", "A", "C", "E"],
            "ead": [1000.0, 2000.0, 3000.0, 1000.0, 2000.0, 3000.0],
            "expected_loss": [5.0, 60.0, 500.0, 8.0, 100.0, 700.0],
            "target_default_90d": [0, 0, 1, 0, 0, 1],
        }
    )
    features = pd.DataFrame(
        {
            "customer_id": [f"CUST_{i:06d}" for i in range(6)],
            "debt_to_income": [0.1, 0.3, 0.9, 0.15, 0.4, 1.0],
            "credit_utilization": [0.1, 0.4, 0.9, 0.2, 0.5, 0.95],
            "late_payments_3m": [0, 0, 2, 0, 1, 3],
        }
    )
    return scored, features


def test_portfolio_summary_basic_fields() -> None:
    scored, features = _sample_scored_and_features()
    summary = portfolio_summary(scored, features)
    assert summary["total_customers"] == 6
    assert summary["portfolio_exposure"] == 12000.0
    assert 0 <= summary["average_pd"] <= 1
    assert summary["risk_band_distribution"]["A"] == 2


def test_monthly_kpi_trend_has_two_months() -> None:
    scored, features = _sample_scored_and_features()
    trend = monthly_kpi_trend(scored, features)
    assert len(trend) == 2
    assert "expected_loss_mom_delta" in trend.columns


def test_compare_segments_returns_both_segments() -> None:
    scored, features = _sample_scored_and_features()
    result = compare_segments(scored, features, "risk_band", "A", "D")
    assert result["A"]["n_customers"] == 2
    assert result["D"]["n_customers"] == 1
    assert result["D"]["average_pd"] > result["A"]["average_pd"]


def _sample_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C1", "C2", "C2", "C2"],
            "snapshot_month": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"] * 2),
            "mob": [0, 1, 2, 0, 1, 2],
            "delinquency_bucket": ["CURRENT", "1-30", "31-60", "CURRENT", "CURRENT", "1-30"],
            "balance": [100.0] * 6,
            "utilization": [0.3] * 6,
            "days_past_due": [0, 15, 45, 0, 0, 15],
        }
    )


def test_build_roll_rate_matrix_rows_sum_to_one_or_zero() -> None:
    matrix = build_roll_rate_matrix(_sample_panel())
    assert list(matrix.columns) == list(DELINQUENCY_BUCKETS)
    row_sums = matrix.sum(axis=1)
    for total in row_sums:
        assert total == 0 or abs(total - 1.0) < 1e-9


def test_cure_and_migration_rates_keys_present() -> None:
    matrix = build_roll_rate_matrix(_sample_panel())
    rates = cure_and_migration_rates(matrix)
    assert "roll_forward_rate_from_current" in rates


def test_build_vintage_curves_and_filter() -> None:
    panel = _sample_panel()
    customers = pd.DataFrame(
        {"customer_id": ["C1", "C2"], "origination_date": pd.to_datetime(["2024-01-31"] * 2)}
    )
    curves = build_vintage_curves(panel, customers)
    assert set(curves["mob"]) == {0, 1, 2}
    assert (curves["bad_rate"] >= 0).all()

    filtered = filter_material_cohorts(curves, min_accounts=2)
    assert (filtered["n_accounts"] >= 0).all()

    summary = cohort_summary(curves, mob_checkpoint=1)
    assert "bad_rate" in summary.columns
