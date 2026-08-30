"""Unit tests for the synthetic credit data generator.

These tests assert the statistical and structural properties the rest of
the pipeline (validation, features, modeling) depends on: uniqueness,
realistic default rate, monotonic risk relationships, and the presence of
deliberately-injected noise (missing values / outliers).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.generate_synthetic_credit_data import (
    GenerationConfig,
    generate_customer_snapshot,
    generate_monthly_performance_panel,
)
from src.data.schema import (
    CUSTOMER_TABLE_COLUMNS,
    DELINQUENCY_BUCKETS,
    PROHIBITED_PROTECTED_ATTRIBUTES,
)


@pytest.fixture(scope="module")
def small_config() -> GenerationConfig:
    return GenerationConfig(
        n_customers=5000,
        random_seed=42,
        start_date="2023-01-01",
        end_date="2025-06-30",
        target_default_rate=0.075,
    )


@pytest.fixture(scope="module")
def customers(small_config: GenerationConfig) -> pd.DataFrame:
    return generate_customer_snapshot(small_config)


def test_expected_columns_present(customers: pd.DataFrame) -> None:
    missing = set(CUSTOMER_TABLE_COLUMNS) - set(customers.columns)
    assert not missing, f"missing expected columns: {missing}"


def test_no_protected_attributes(customers: pd.DataFrame) -> None:
    lower_cols = {c.lower() for c in customers.columns}
    banned = set(PROHIBITED_PROTECTED_ATTRIBUTES) & lower_cols
    assert not banned, f"prohibited protected attributes present: {banned}"


def test_customer_id_is_unique_and_synthetic(customers: pd.DataFrame) -> None:
    assert customers["customer_id"].is_unique
    assert customers["customer_id"].str.match(r"^CUST_\d{6}$").all()


def test_target_is_binary(customers: pd.DataFrame) -> None:
    assert set(customers["target_default_90d"].unique()) <= {0, 1}


def test_default_rate_is_realistic(customers: pd.DataFrame) -> None:
    rate = customers["target_default_90d"].mean()
    # Real unsecured-credit portfolios typically sit well below 20%; the
    # generator targets ~7.5% by construction, allow reasonable sampling noise.
    assert 0.02 <= rate <= 0.15, f"unrealistic default rate: {rate}"


def test_default_rate_increases_with_debt_to_income(customers: pd.DataFrame) -> None:
    quartiles = pd.qcut(customers["debt_to_income"], 4, duplicates="drop")
    rates = customers.groupby(quartiles, observed=True)["target_default_90d"].mean()
    assert rates.is_monotonic_increasing


def test_default_rate_increases_with_utilization(customers: pd.DataFrame) -> None:
    quartiles = pd.qcut(customers["credit_utilization"], 4, duplicates="drop")
    rates = customers.groupby(quartiles, observed=True)["target_default_90d"].mean()
    assert rates.is_monotonic_increasing


def test_previous_default_flag_increases_risk(customers: pd.DataFrame) -> None:
    rates = customers.groupby("previous_default_flag")["target_default_90d"].mean()
    assert rates.loc[1] > rates.loc[0]


def test_missing_values_injected(customers: pd.DataFrame) -> None:
    assert customers["declared_expenses"].isna().mean() > 0
    assert customers["average_monthly_spend"].isna().mean() > 0


def test_generation_is_deterministic_given_seed(small_config: GenerationConfig) -> None:
    a = generate_customer_snapshot(small_config)
    b = generate_customer_snapshot(small_config)
    pd.testing.assert_frame_equal(a, b)


def test_monthly_performance_panel_structure(
    customers: pd.DataFrame, small_config: GenerationConfig
) -> None:
    panel = generate_monthly_performance_panel(customers, small_config, max_mob=6)
    assert set(panel["delinquency_bucket"].unique()) <= set(DELINQUENCY_BUCKETS)
    assert (panel["mob"] >= 0).all()
    assert panel["mob"].max() <= 6
    # Every customer must have at least a MOB-0 record.
    assert panel.groupby("customer_id")["mob"].min().max() == 0


def test_monthly_performance_panel_no_negative_values(
    customers: pd.DataFrame, small_config: GenerationConfig
) -> None:
    panel = generate_monthly_performance_panel(customers, small_config, max_mob=6)
    assert (panel["balance"] >= 0).all()
    assert (panel["days_past_due"] >= 0).all()


@pytest.mark.parametrize("n", [0, 1])
def test_generate_handles_tiny_populations(n: int) -> None:
    if n == 0:
        pytest.skip("n=0 is not a supported/meaningful configuration")
    cfg = GenerationConfig(
        n_customers=n,
        random_seed=1,
        start_date="2023-01-01",
        end_date="2023-06-30",
    )
    df = generate_customer_snapshot(cfg)
    assert len(df) == n
    assert not np.isnan(df["monthly_income"].iloc[0])
