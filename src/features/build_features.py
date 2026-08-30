"""Feature engineering pipeline for the credit risk model.

Turns the raw synthetic customer snapshot into a modeling-ready dataset:
adds a small set of engineered risk indicators on top of the raw fields,
and assigns each row to a temporal split (train / validation / test) based
on `observation_date` so that model evaluation respects out-of-time (OOT)
validation instead of a random split (see docs/TEMPORAL_VALIDATION.md).

Every derived feature is documented in FEATURE_DICTIONARY.md - keep the two
in sync when adding a new feature here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.config import PROJECT_ROOT, get_settings
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

CONFIG_PATH = PROJECT_ROOT / "configs" / "project_config.yaml"

ENGINEERED_FEATURE_COLUMNS: tuple[str, ...] = (
    "payment_stress_index",
    "behavioral_deterioration_index",
    "recent_inquiry_intensity",
    "spend_to_income",
    "utilization_gap_to_limit",
    "is_thin_file",
)


def load_temporal_split_config(config_path: Path = CONFIG_PATH) -> dict[str, pd.Timestamp]:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    split_cfg = raw["temporal_split"]
    return {k: pd.Timestamp(v) for k, v in split_cfg.items()}


def assign_temporal_split(
    observation_date: pd.Series, split_dates: dict[str, pd.Timestamp]
) -> pd.Series:
    """Assign each row to train/validation/test based on `observation_date`.

    This is deliberately NOT `train_test_split(random_state=...)`: rows are
    partitioned strictly by calendar time so the test set represents a true
    out-of-time (OOT) sample and the model's temporal generalization can be
    measured honestly (see docs/TEMPORAL_VALIDATION.md for the leakage
    discussion).
    """
    conditions = [
        observation_date.between(split_dates["train_start"], split_dates["train_end"]),
        observation_date.between(split_dates["validation_start"], split_dates["validation_end"]),
        observation_date.between(split_dates["test_start"], split_dates["test_end"]),
    ]
    choices = ["train", "validation", "test"]
    return pd.Series(
        np.select(conditions, choices, default="excluded"), index=observation_date.index
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered risk-indicator columns to the raw customer snapshot.

    All engineered features are built exclusively from already-present
    observable fields (never from the target), so there is no target
    leakage. NaNs in raw inputs propagate to NaN in derived features rather
    than being silently imputed here - imputation is a modeling-stage
    decision (see src/models/train.py).
    """
    out = df.copy()

    # Payment Stress Index: combines delinquency, indebtedness and
    # utilization into a single bounded [0, 1]-ish composite indicator.
    late_component = np.clip(out["late_payments_12m"] / 6.0, 0, 1)
    dti_component = np.clip(out["debt_to_income"] / 1.5, 0, 1)
    utilization_component = np.clip(out["credit_utilization"], 0, 1)
    out["payment_stress_index"] = (
        0.4 * late_component + 0.35 * dti_component + 0.25 * utilization_component
    ).round(4)

    # Behavioral Deterioration Index: aggregates the three directional
    # trend signals generated upstream (balance/utilization/delinquency).
    out["behavioral_deterioration_index"] = (
        0.4 * out["delinquency_trend"]
        + 0.35 * out["utilization_trend"]
        + 0.25 * out["balance_trend"]
    ).round(4)

    # Recent credit-seeking intensity, normalized by how long the account
    # has existed so long-tenured and brand-new customers are comparable.
    out["recent_inquiry_intensity"] = (
        out["number_of_recent_credit_inquiries"] / np.maximum(out["account_tenure_months"], 1)
    ).round(4)

    out["spend_to_income"] = (
        out["average_monthly_spend"] / out["monthly_income"].replace(0, np.nan)
    ).round(4)

    out["utilization_gap_to_limit"] = (1 - np.clip(out["credit_utilization"], 0, 1)).round(4)

    # Thin-file flag: very short tenure and few open accounts, a segment
    # that typically needs different underwriting treatment.
    out["is_thin_file"] = (
        (out["account_tenure_months"] <= 6) & (out["number_of_open_accounts"] <= 1)
    ).astype(int)

    return out


def build_feature_dataset(customers: pd.DataFrame) -> pd.DataFrame:
    df = engineer_features(customers)
    split_dates = load_temporal_split_config()
    df["split"] = assign_temporal_split(df["observation_date"], split_dates)
    logger.info(
        "temporal_split_assigned",
        **df["split"].value_counts().to_dict(),
    )
    return df


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    raw_path = settings.data_path / "raw" / "customer_portfolio.parquet"
    customers = pd.read_parquet(raw_path)

    features = build_feature_dataset(customers)

    output_dir = settings.data_path / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "credit_features.parquet"
    features.to_parquet(output_path, index=False)

    logger.info(
        "feature_dataset_written",
        path=str(output_path.relative_to(PROJECT_ROOT)),
        n_rows=len(features),
        n_columns=len(features.columns),
    )


if __name__ == "__main__":
    main()
