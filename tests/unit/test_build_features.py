"""Unit tests for src.features.build_features."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.generate_synthetic_credit_data import GenerationConfig, generate_customer_snapshot
from src.features.build_features import (
    ENGINEERED_FEATURE_COLUMNS,
    assign_temporal_split,
    build_feature_dataset,
    engineer_features,
)


@pytest.fixture(scope="module")
def customers() -> pd.DataFrame:
    cfg = GenerationConfig(
        n_customers=4000, random_seed=11, start_date="2023-01-01", end_date="2025-06-30"
    )
    return generate_customer_snapshot(cfg)


def test_engineered_columns_present(customers: pd.DataFrame) -> None:
    out = engineer_features(customers)
    for col in ENGINEERED_FEATURE_COLUMNS:
        assert col in out.columns


def test_engineered_features_do_not_use_target(customers: pd.DataFrame) -> None:
    without_target = customers.drop(columns=["target_default_90d"])
    out = engineer_features(without_target)
    for col in ENGINEERED_FEATURE_COLUMNS:
        assert col in out.columns  # computable without ever touching the label


def test_payment_stress_index_bounded(customers: pd.DataFrame) -> None:
    out = engineer_features(customers)
    assert out["payment_stress_index"].between(0, 1).all()


def test_thin_file_flag_is_binary(customers: pd.DataFrame) -> None:
    out = engineer_features(customers)
    assert set(out["is_thin_file"].unique()) <= {0, 1}


def test_assign_temporal_split_is_disjoint_and_ordered() -> None:
    dates = pd.to_datetime(["2023-03-01", "2024-08-01", "2025-02-01", "2022-01-01", "2026-01-01"])
    split_dates = {
        "train_start": pd.Timestamp("2023-01-01"),
        "train_end": pd.Timestamp("2024-06-30"),
        "validation_start": pd.Timestamp("2024-07-01"),
        "validation_end": pd.Timestamp("2024-12-31"),
        "test_start": pd.Timestamp("2025-01-01"),
        "test_end": pd.Timestamp("2025-06-30"),
    }
    result = assign_temporal_split(pd.Series(dates), split_dates)
    assert list(result) == ["train", "validation", "test", "excluded", "excluded"]


def test_build_feature_dataset_adds_split_column(customers: pd.DataFrame) -> None:
    df = build_feature_dataset(customers)
    assert "split" in df.columns
    assert set(df["split"].unique()) <= {"train", "validation", "test", "excluded"}
    # No leakage: train rows must all have observation_date before any test row's minimum.
    if (df["split"] == "train").any() and (df["split"] == "test").any():
        assert (
            df.loc[df["split"] == "train", "observation_date"].max()
            <= df.loc[df["split"] == "test", "observation_date"].min()
        )
