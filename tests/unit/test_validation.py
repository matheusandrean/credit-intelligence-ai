"""Unit tests for src.data.validation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.generate_synthetic_credit_data import GenerationConfig, generate_customer_snapshot
from src.data.validation import run_data_quality_checks


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    cfg = GenerationConfig(
        n_customers=3000, random_seed=7, start_date="2023-01-01", end_date="2025-06-30"
    )
    return generate_customer_snapshot(cfg)


def test_clean_generated_data_passes(clean_df: pd.DataFrame) -> None:
    report = run_data_quality_checks(clean_df)
    assert report.schema_valid
    assert report.duplicate_customer_ids == 0
    assert not report.protected_attributes_found
    assert not report.consistency_failures
    assert report.passed


def test_detects_duplicate_customer_ids(clean_df: pd.DataFrame) -> None:
    dup = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)
    report = run_data_quality_checks(dup)
    assert report.duplicate_customer_ids == 1
    assert not report.passed


def test_detects_protected_attribute_column(clean_df: pd.DataFrame) -> None:
    tainted = clean_df.copy()
    tainted["gender"] = "unknown"
    report = run_data_quality_checks(tainted)
    assert "gender" in report.protected_attributes_found
    assert not report.passed


def test_detects_late_payment_inconsistency(clean_df: pd.DataFrame) -> None:
    broken = clean_df.copy()
    broken.loc[0, "late_payments_3m"] = broken.loc[0, "late_payments_6m"] + 5
    report = run_data_quality_checks(broken)
    assert any("late_payments_3m" in f for f in report.consistency_failures)
    assert not report.passed


def test_detects_out_of_range_target(clean_df: pd.DataFrame) -> None:
    broken = clean_df.copy()
    broken.loc[0, "target_default_90d"] = 2
    report = run_data_quality_checks(broken)
    assert not report.schema_valid
    assert not report.passed


def test_null_rate_summary_reports_known_missing_columns(clean_df: pd.DataFrame) -> None:
    report = run_data_quality_checks(clean_df)
    assert report.null_rates["declared_expenses"] > 0
