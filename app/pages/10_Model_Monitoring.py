"""Model Monitoring: PSI drift for features and the predicted PD distribution."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.theme import (
    configure_page,
    demo_data_disclaimer,
    load_features,
    load_scored_portfolio,
    require_artifacts,
)
from src.monitoring.drift import feature_drift_report, model_performance_over_time

configure_page("Model Monitoring", icon="\U0001f4e1")
st.title("Model Monitoring")
demo_data_disclaimer()
st.caption(
    "PSI thresholds (Stable < 0.10, Monitor 0.10-0.25, Potential significant drift > 0.25) "
    "are demonstrative conventions - see GOVERNANCE.md."
)

if not require_artifacts():
    st.stop()

features = load_features()
scored = load_scored_portfolio()

st.subheader("Feature Drift (Train vs OOT Test)")
train = features[features["split"] == "train"]
test = features[features["split"] == "test"]
candidate_features = [
    "debt_to_income",
    "credit_utilization",
    "monthly_income",
    "late_payments_12m",
    "payment_stress_index",
    "behavioral_deterioration_index",
    "financial_stability_index",
]
report = feature_drift_report(train, test, candidate_features)


def _status_color(status: str) -> str:
    return {
        "Stable": "#1B7A4C",
        "Monitor": "#D6A419",
        "Potential significant drift": "#B3261E",
    }.get(status, "#999")


fig = px.bar(
    report,
    x="feature",
    y="psi",
    color="status",
    color_discrete_map={
        "Stable": "#1B7A4C",
        "Monitor": "#D6A419",
        "Potential significant drift": "#B3261E",
    },
)
fig.update_layout(height=400)
st.plotly_chart(fig, width="stretch")
st.dataframe(report, width="stretch")

st.markdown("---")
st.subheader("Model Performance Over Time")
perf = model_performance_over_time(scored)
if not perf.empty:
    fig2 = px.line(perf, x="month", y="roc_auc", markers=True)
    fig2.update_layout(height=380, yaxis_range=[0.5, 1.0], title="ROC-AUC by month")
    st.plotly_chart(fig2, width="stretch")

    fig3 = px.line(perf, x="month", y="psi_vs_first_month", markers=True)
    fig3.add_hline(y=0.10, line_dash="dash", line_color="#D6A419", annotation_text="Monitor")
    fig3.add_hline(y=0.25, line_dash="dash", line_color="#B3261E", annotation_text="Significant")
    fig3.update_layout(height=380, title="Predicted PD PSI vs first available month")
    st.plotly_chart(fig3, width="stretch")

    st.dataframe(perf, width="stretch")
else:
    st.info("Not enough monthly history to compute performance-over-time.")
