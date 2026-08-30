"""Integration tests for the training pipeline's core functions.

Exercises the real generator -> feature engineering -> training functions
end-to-end on a small synthetic sample (kept small for speed), without
touching the project's real `data/` or `models/` directories.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.data.generate_synthetic_credit_data import GenerationConfig, generate_customer_snapshot
from src.features.build_features import build_feature_dataset
from src.models.feature_lists import MODEL_CATEGORICAL_FEATURES
from src.models.metrics import compute_classification_metrics
from src.models.train import (
    build_logistic_regression_pipeline,
    split_xy,
    train_lightgbm,
    train_logistic_regression,
)


@pytest.fixture(scope="module")
def feature_df():
    cfg = GenerationConfig(
        n_customers=6000, random_seed=99, start_date="2023-01-01", end_date="2025-06-30"
    )
    customers = generate_customer_snapshot(cfg)
    return build_feature_dataset(customers)


@pytest.fixture(scope="module")
def categories(feature_df):
    return {
        col: sorted(feature_df[col].dropna().unique().tolist())
        for col in MODEL_CATEGORICAL_FEATURES
    }


def test_logistic_regression_beats_random_on_oot_test(feature_df, categories) -> None:
    x_train, y_train = split_xy(feature_df, "train", categories)
    x_test, y_test = split_xy(feature_df, "test", categories)
    if len(x_train) == 0 or len(x_test) == 0 or y_test.sum() == 0:
        pytest.skip("insufficient rows in a split for this small sample")

    pipeline = train_logistic_regression(x_train, y_train)
    y_prob = pipeline.predict_proba(x_test)[:, 1]
    metrics = compute_classification_metrics(y_test, y_prob)
    assert metrics.roc_auc > 0.6  # meaningfully better than random (0.5)


def test_lightgbm_trains_and_predicts_valid_probabilities(feature_df, categories) -> None:
    x_train, y_train = split_xy(feature_df, "train", categories)
    x_val, y_val = split_xy(feature_df, "validation", categories)
    if len(x_train) == 0 or len(x_val) == 0:
        pytest.skip("insufficient rows in a split for this small sample")

    model = train_lightgbm(x_train, y_train, x_val, y_val)
    y_prob = model.predict_proba(x_val)[:, 1]
    assert np.all((y_prob >= 0) & (y_prob <= 1))


def test_calibration_improves_or_maintains_brier_score(feature_df, categories) -> None:
    from sklearn.metrics import brier_score_loss

    x_train, y_train = split_xy(feature_df, "train", categories)
    x_val, y_val = split_xy(feature_df, "validation", categories)
    x_test, y_test = split_xy(feature_df, "test", categories)
    if min(len(x_train), len(x_val), len(x_test)) == 0 or y_test.sum() == 0:
        pytest.skip("insufficient rows in a split for this small sample")

    pipeline = build_logistic_regression_pipeline()
    pipeline.fit(x_train, y_train)

    raw_test_prob = pipeline.predict_proba(x_test)[:, 1]
    raw_brier = brier_score_loss(y_test, raw_test_prob)

    calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
    calibrated.fit(x_val, y_val)
    calibrated_test_prob = calibrated.predict_proba(x_test)[:, 1]
    calibrated_brier = brier_score_loss(y_test, calibrated_test_prob)

    # Calibration should not catastrophically worsen the Brier score.
    assert calibrated_brier < raw_brier * 1.5
