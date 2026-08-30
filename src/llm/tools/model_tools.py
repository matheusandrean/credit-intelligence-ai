"""Model-introspection LLM tools: feature importance, per-customer SHAP,
model performance metrics, and drift detection."""

from __future__ import annotations

import json
from typing import Any

from src.llm.schemas import (
    DetectDriftInput,
    GetCustomerShapInput,
    GetFeatureImportanceInput,
    GetModelMetricsInput,
)
from src.llm.tools.context import ToolContext
from src.monitoring.drift import feature_drift_report
from src.utils.config import PROJECT_ROOT


def tool_get_feature_importance(
    input_data: GetFeatureImportanceInput, ctx: ToolContext
) -> dict[str, Any]:
    categories = ctx.model_metadata["categories"]
    from src.models.train import split_xy

    x_test, _ = split_xy(ctx.features, "test", categories)
    if x_test.empty:
        x_test, _ = split_xy(ctx.features, "train", categories)
    sample = x_test.sample(min(500, len(x_test)), random_state=1)
    importance = ctx.shap_explainer.global_importance(sample)
    top = importance.head(input_data.top_n)
    return {
        "ok": True,
        "data": {
            "champion_model": ctx.model_metadata["champion_model"],
            "top_features": top.to_dict(orient="records"),
        },
    }


def tool_get_customer_shap(input_data: GetCustomerShapInput, ctx: ToolContext) -> dict[str, Any]:
    row = ctx.get_customer_row(input_data.customer_id)
    if row is None:
        return {"ok": False, "error": f"Customer {input_data.customer_id} not found."}

    from src.models.feature_lists import MODEL_FEATURES

    x_row = row[list(MODEL_FEATURES)].copy()
    categories = ctx.model_metadata["categories"]
    for col, cats in categories.items():
        import pandas as pd

        x_row[col] = pd.Categorical(x_row[col], categories=cats)

    scored_row = ctx.scored_portfolio[ctx.scored_portfolio["customer_id"] == input_data.customer_id]
    predicted_pd = (
        float(scored_row.iloc[0]["pd"])
        if not scored_row.empty
        else float(ctx.champion_model.predict_proba(x_row)[0, 1])
    )

    explanation = ctx.shap_explainer.explain_customer(
        input_data.customer_id, x_row, predicted_pd=predicted_pd, top_k=input_data.top_k
    )
    return {
        "ok": True,
        "data": {
            "customer_id": explanation.customer_id,
            "predicted_pd": explanation.predicted_pd,
            "top_risk_increasing": explanation.top_risk_increasing,
            "top_risk_reducing": explanation.top_risk_reducing,
        },
    }


def tool_get_model_metrics(_input: GetModelMetricsInput, ctx: ToolContext) -> dict[str, Any]:
    comparison_path = PROJECT_ROOT / "reports" / "model_comparison.json"
    calibration_path = PROJECT_ROOT / "reports" / "calibration_report.json"
    data: dict[str, Any] = {"champion_model": ctx.model_metadata["champion_model"]}
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        champion = ctx.model_metadata["champion_model"]
        data["champion_test_metrics"] = comparison.get(champion, {}).get("test")
        data["all_models_test_metrics"] = {k: v.get("test") for k, v in comparison.items()}
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        data["calibration_method"] = calibration.get("calibration_method")
        data["raw_brier_test"] = calibration.get("raw_brier_test")
        data["calibrated_brier_test"] = calibration.get("calibrated_brier_test")
    return {"ok": True, "data": data}


def tool_detect_drift(input_data: DetectDriftInput, ctx: ToolContext) -> dict[str, Any]:
    train = ctx.features[ctx.features["split"] == "train"]
    test = ctx.features[ctx.features["split"] == "test"]
    if train.empty or test.empty:
        return {"ok": False, "error": "Insufficient train/test data to compute drift."}

    default_features = [
        "debt_to_income",
        "credit_utilization",
        "monthly_income",
        "late_payments_12m",
        "payment_stress_index",
        "behavioral_deterioration_index",
    ]
    features_to_check = input_data.features or default_features
    report = feature_drift_report(train, test, features_to_check)

    from src.monitoring.drift import model_performance_over_time

    perf = model_performance_over_time(ctx.scored_portfolio)
    latest_psi = (
        perf.iloc[-1][["month", "psi_vs_first_month", "psi_status"]].to_dict()
        if not perf.empty
        else None
    )

    return {
        "ok": True,
        "data": {
            "feature_drift": report.to_dict(orient="records"),
            "predicted_pd_drift_latest_month": latest_psi,
        },
    }
