"""Portfolio-level KPIs and segment comparisons.

Consumes the scored portfolio (src/risk/scorebook.py output) joined with the
engineered feature dataset to produce the executive/MIS-style indicators
used across the dashboard: exposure, risk, loss and behavior metrics, with
month-over-month (MoM) and quarter-over-quarter (QoQ) deltas.
"""

from __future__ import annotations

import pandas as pd


def portfolio_summary(scored: pd.DataFrame, features: pd.DataFrame) -> dict:
    """Single-period portfolio KPI snapshot."""
    merged = scored.merge(
        features[["customer_id", "debt_to_income", "credit_utilization", "late_payments_3m"]],
        on="customer_id",
        how="left",
    )
    n_customers = len(merged)
    high_risk = merged["risk_band"].isin(["D", "E"]).mean() if n_customers else 0.0
    delinquent = (merged["late_payments_3m"] > 0).mean() if n_customers else 0.0

    return {
        "total_customers": int(n_customers),
        "portfolio_exposure": float(merged["ead"].sum()),
        "average_pd": float(merged["pd"].mean()),
        "expected_loss": float(merged["expected_loss"].sum()),
        "expected_loss_rate": (
            float(merged["expected_loss"].sum() / merged["ead"].sum())
            if merged["ead"].sum() > 0
            else 0.0
        ),
        "observed_default_rate": float(merged["target_default_90d"].mean()),
        "delinquency_rate_3m": float(delinquent),
        "high_risk_population_pct": float(high_risk),
        "average_dti": float(merged["debt_to_income"].mean()),
        "average_utilization": float(merged["credit_utilization"].mean()),
        "average_score": float(merged["score"].mean()),
        "risk_band_distribution": merged["risk_band"].value_counts().to_dict(),
    }


def monthly_kpi_trend(scored: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Portfolio KPIs aggregated by observation month, for MoM/QoQ trending."""
    merged = scored.merge(
        features[["customer_id", "debt_to_income", "credit_utilization"]],
        on="customer_id",
        how="left",
    )
    merged["month"] = pd.to_datetime(merged["observation_date"]).dt.to_period("M")

    grouped = merged.groupby("month").agg(
        total_customers=("customer_id", "count"),
        portfolio_exposure=("ead", "sum"),
        average_pd=("pd", "mean"),
        expected_loss=("expected_loss", "sum"),
        observed_default_rate=("target_default_90d", "mean"),
        average_dti=("debt_to_income", "mean"),
        average_utilization=("credit_utilization", "mean"),
        average_score=("score", "mean"),
        high_risk_population_pct=("risk_band", lambda s: s.isin(["D", "E"]).mean()),
    )
    grouped = grouped.sort_index()
    grouped["expected_loss_mom_delta"] = grouped["expected_loss"].diff()
    grouped["average_pd_mom_delta"] = grouped["average_pd"].diff()
    grouped["quarter"] = grouped.index.asfreq("Q")
    quarterly = grouped.groupby("quarter")[["average_pd", "expected_loss"]].mean()
    grouped["average_pd_qoq_delta"] = grouped["quarter"].map(quarterly["average_pd"].diff())
    grouped = grouped.reset_index()
    grouped["month"] = grouped["month"].astype(str)
    return grouped


def compare_segments(
    scored: pd.DataFrame,
    features: pd.DataFrame,
    segment_column: str,
    segment_a: str,
    segment_b: str,
) -> dict:
    """Compare two segments (e.g. risk bands, age bands) on key indicators."""
    merged = scored.merge(features, on="customer_id", how="left", suffixes=("", "_feat"))

    def _segment_stats(value: str) -> dict:
        subset = merged[merged[segment_column] == value]
        if subset.empty:
            return {"n_customers": 0}
        return {
            "n_customers": int(len(subset)),
            "average_pd": float(subset["pd"].mean()),
            "average_score": float(subset["score"].mean()),
            "expected_loss": float(subset["expected_loss"].sum()),
            "average_dti": float(subset["debt_to_income"].mean()),
            "average_utilization": float(subset["credit_utilization"].mean()),
            "observed_default_rate": float(subset["target_default_90d"].mean()),
        }

    return {
        "segment_column": segment_column,
        segment_a: _segment_stats(segment_a),
        segment_b: _segment_stats(segment_b),
    }
