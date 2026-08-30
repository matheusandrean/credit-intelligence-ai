"""Portfolio-query LLM tools: summary, customer profile, segment comparison,
default rate, expected loss, and guarded text-to-SQL.

Every function here takes a validated Pydantic input and the shared
`ToolContext`, and returns a plain JSON-serializable dict. These are the
only way the LLM agent can touch portfolio data - see RESPONSIBLE_AI.md.
"""

from __future__ import annotations

from typing import Any

from src.analytics.portfolio import compare_segments, portfolio_summary
from src.llm.schemas import (
    CalculateDefaultRateInput,
    CalculateExpectedLossInput,
    CompareSegmentsInput,
    CustomerRiskProfileInput,
    PortfolioSummaryInput,
    QueryPortfolioInput,
)
from src.llm.tools.context import ToolContext
from src.llm.tools.json_utils import to_json_safe
from src.llm.tools.sql_guard import UnsafeSqlError, validate_select_only


def tool_portfolio_summary(_input: PortfolioSummaryInput, ctx: ToolContext) -> dict[str, Any]:
    summary = portfolio_summary(ctx.scored_portfolio, ctx.features)
    return {"ok": True, "data": summary}


def tool_customer_risk_profile(
    input_data: CustomerRiskProfileInput, ctx: ToolContext
) -> dict[str, Any]:
    row = ctx.scored_portfolio[ctx.scored_portfolio["customer_id"] == input_data.customer_id]
    if row.empty:
        return {"ok": False, "error": f"Customer {input_data.customer_id} not found."}
    feat_row = ctx.features[ctx.features["customer_id"] == input_data.customer_id]
    r = row.iloc[0]
    profile = {
        "customer_id": input_data.customer_id,
        "pd": float(r["pd"]),
        "score": float(r["score"]),
        "risk_band": str(r["risk_band"]),
        "expected_loss": float(r["expected_loss"]),
        "exposure_at_default": float(r["ead"]),
        "observed_default_90d": bool(r["target_default_90d"]),
    }
    if not feat_row.empty:
        f = feat_row.iloc[0]
        profile["behavior"] = {
            "monthly_income": float(f["monthly_income"]) if pd_notna(f["monthly_income"]) else None,
            "debt_to_income": float(f["debt_to_income"]) if pd_notna(f["debt_to_income"]) else None,
            "credit_utilization": (
                float(f["credit_utilization"]) if pd_notna(f["credit_utilization"]) else None
            ),
            "late_payments_12m": int(f["late_payments_12m"]),
            "previous_default_flag": int(f["previous_default_flag"]),
            "account_tenure_months": int(f["account_tenure_months"]),
        }
    return {"ok": True, "data": profile}


def pd_notna(value: Any) -> bool:
    import pandas as pd

    return pd.notna(value)


def tool_compare_segments(input_data: CompareSegmentsInput, ctx: ToolContext) -> dict[str, Any]:
    if input_data.segment_column not in ctx.scored_portfolio.columns and (
        input_data.segment_column not in ctx.features.columns
    ):
        return {"ok": False, "error": f"Unknown segment column: {input_data.segment_column}"}
    result = compare_segments(
        ctx.scored_portfolio,
        ctx.features,
        input_data.segment_column,
        input_data.segment_a,
        input_data.segment_b,
    )
    return {"ok": True, "data": result}


def tool_calculate_default_rate(
    input_data: CalculateDefaultRateInput, ctx: ToolContext
) -> dict[str, Any]:
    df = ctx.scored_portfolio
    if input_data.segment_column:
        merged = df.merge(ctx.features, on="customer_id", how="left", suffixes=("", "_feat"))
        if input_data.segment_column not in merged.columns:
            return {"ok": False, "error": f"Unknown segment column: {input_data.segment_column}"}
        breakdown = merged.groupby(input_data.segment_column)["target_default_90d"].mean().to_dict()
        return {
            "ok": True,
            "data": {
                "overall_default_rate": float(df["target_default_90d"].mean()),
                "by_segment": {str(k): float(v) for k, v in breakdown.items()},
            },
        }
    return {"ok": True, "data": {"overall_default_rate": float(df["target_default_90d"].mean())}}


def tool_calculate_expected_loss(
    input_data: CalculateExpectedLossInput, ctx: ToolContext
) -> dict[str, Any]:
    df = ctx.scored_portfolio
    total_el = float(df["expected_loss"].sum())
    total_exposure = float(df["ead"].sum())
    result: dict[str, Any] = {
        "total_expected_loss": total_el,
        "total_exposure": total_exposure,
        "expected_loss_rate": total_el / total_exposure if total_exposure > 0 else 0.0,
    }
    if input_data.segment_column:
        merged = df.merge(ctx.features, on="customer_id", how="left", suffixes=("", "_feat"))
        if input_data.segment_column not in merged.columns:
            return {"ok": False, "error": f"Unknown segment column: {input_data.segment_column}"}
        breakdown = merged.groupby(input_data.segment_column)["expected_loss"].sum().to_dict()
        result["by_segment"] = {str(k): float(v) for k, v in breakdown.items()}
    return {"ok": True, "data": result}


def tool_query_portfolio(input_data: QueryPortfolioInput, ctx: ToolContext) -> dict[str, Any]:
    try:
        safe_sql = validate_select_only(input_data.sql)
    except UnsafeSqlError as exc:
        return {"ok": False, "error": f"Rejected unsafe SQL: {exc}"}

    try:
        result_df = ctx.duckdb_connection.execute(safe_sql).df()
    except Exception as exc:  # noqa: BLE001 - surfaced as a tool error, not a crash
        return {"ok": False, "error": f"SQL execution failed: {exc}"}

    return {
        "ok": True,
        "data": {
            "columns": list(result_df.columns),
            "rows": to_json_safe(result_df.head(200).to_dict(orient="records")),
            "n_rows_returned": len(result_df),
        },
    }
