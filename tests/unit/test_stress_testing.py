"""Unit tests for src.risk.stress_testing (shock mechanics only; the
scenario-runner functions that load the trained model are covered by the
integration test, guarded by artifact availability)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.risk.stress_testing import apply_shock, load_stress_scenarios


@pytest.fixture
def sample_row() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "monthly_income": [8000.0],
            "declared_expenses": [3000.0],
            "average_monthly_spend": [2000.0],
            "debt_to_income": [0.30],
            "installment_to_income": [0.10],
            "credit_utilization": [0.60],
            "credit_limit": [10000.0],
            "revolving_balance": [6000.0],
            "late_payments_12m": [1],
            "account_tenure_months": [24],
            "number_of_open_accounts": [3],
            "number_of_recent_credit_inquiries": [2],
            "delinquency_trend": [0.05],
            "utilization_trend": [0.02],
            "balance_trend": [0.01],
        }
    )


def test_income_shock_reduces_income_and_raises_dti(sample_row: pd.DataFrame) -> None:
    shocked = apply_shock(sample_row, income_shock_pct=-0.10)
    assert shocked["monthly_income"].iloc[0] == pytest.approx(7200.0)
    assert shocked["debt_to_income"].iloc[0] > sample_row["debt_to_income"].iloc[0]


def test_expense_shock_raises_expenses(sample_row: pd.DataFrame) -> None:
    shocked = apply_shock(sample_row, expense_shock_pct=0.10)
    assert shocked["declared_expenses"].iloc[0] == pytest.approx(3300.0)


def test_utilization_shock_raises_utilization_and_balance(sample_row: pd.DataFrame) -> None:
    shocked = apply_shock(sample_row, utilization_shock_pp=10.0)
    assert shocked["credit_utilization"].iloc[0] == pytest.approx(0.70)
    assert shocked["revolving_balance"].iloc[0] == pytest.approx(7000.0)


def test_zero_shock_is_a_no_op_on_key_fields(sample_row: pd.DataFrame) -> None:
    shocked = apply_shock(sample_row)
    assert shocked["monthly_income"].iloc[0] == pytest.approx(sample_row["monthly_income"].iloc[0])
    assert shocked["credit_utilization"].iloc[0] == pytest.approx(
        sample_row["credit_utilization"].iloc[0]
    )


def test_engineered_features_recomputed_after_shock(sample_row: pd.DataFrame) -> None:
    shocked = apply_shock(sample_row, utilization_shock_pp=20.0)
    assert "payment_stress_index" in shocked.columns
    assert shocked["payment_stress_index"].iloc[0] >= 0


def test_load_stress_scenarios_has_expected_names() -> None:
    scenarios = load_stress_scenarios()
    assert {"baseline", "mild", "moderate", "severe"} <= set(scenarios.keys())
    assert scenarios["severe"].income_shock_pct < scenarios["mild"].income_shock_pct
