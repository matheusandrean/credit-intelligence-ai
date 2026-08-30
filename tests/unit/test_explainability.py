"""Unit tests for src.models.explainability using a freshly trained small model
(no dependency on pre-built project artifacts)."""

from __future__ import annotations

import pytest

from src.data.generate_synthetic_credit_data import GenerationConfig, generate_customer_snapshot
from src.features.build_features import build_feature_dataset
from src.models.explainability import ShapExplainer
from src.models.feature_lists import MODEL_CATEGORICAL_FEATURES
from src.models.train import build_logistic_regression_pipeline, split_xy


@pytest.fixture(scope="module")
def trained_context():
    cfg = GenerationConfig(
        n_customers=4000, random_seed=5, start_date="2023-01-01", end_date="2025-06-30"
    )
    customers = generate_customer_snapshot(cfg)
    df = build_feature_dataset(customers)
    categories = {
        col: sorted(df[col].dropna().unique().tolist()) for col in MODEL_CATEGORICAL_FEATURES
    }
    x_train, y_train = split_xy(df, "train", categories)
    if len(x_train) < 100:
        pytest.skip("insufficient training rows in this small sample")
    pipeline = build_logistic_regression_pipeline()
    pipeline.fit(x_train, y_train)
    return pipeline, x_train, df, categories


def test_shap_explainer_builds_for_logistic_regression(trained_context) -> None:
    pipeline, x_train, _, _ = trained_context
    background = x_train.sample(min(50, len(x_train)), random_state=1)
    explainer = ShapExplainer("logistic_regression", pipeline, background)
    assert len(explainer.feature_names) > 0


def test_explain_customer_returns_readable_values(trained_context) -> None:
    pipeline, x_train, _, _ = trained_context
    background = x_train.sample(min(50, len(x_train)), random_state=1)
    explainer = ShapExplainer("logistic_regression", pipeline, background)

    row = x_train.iloc[[0]]
    pd_hat = float(pipeline.predict_proba(row)[0, 1])
    explanation = explainer.explain_customer("CUST_TEST", row, predicted_pd=pd_hat)

    assert explanation.customer_id == "CUST_TEST"
    assert len(explanation.top_risk_increasing) <= 5
    assert len(explanation.top_risk_reducing) <= 5
    for factor in explanation.top_risk_increasing:
        assert factor["shap_value"] > 0
        assert "__" not in factor["feature"]  # no leaked preprocessing prefixes
    for factor in explanation.top_risk_reducing:
        assert factor["shap_value"] < 0


def test_global_importance_sums_to_positive_values(trained_context) -> None:
    pipeline, x_train, _, _ = trained_context
    background = x_train.sample(min(50, len(x_train)), random_state=1)
    explainer = ShapExplainer("logistic_regression", pipeline, background)

    sample = x_train.sample(min(100, len(x_train)), random_state=2)
    importance = explainer.global_importance(sample)
    assert (importance["mean_abs_shap"] >= 0).all()
    assert importance.iloc[0]["mean_abs_shap"] >= importance.iloc[-1]["mean_abs_shap"]
    assert not importance["feature"].str.contains("__").any()
