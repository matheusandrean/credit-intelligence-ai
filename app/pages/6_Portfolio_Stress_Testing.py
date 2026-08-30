"""Portfolio Stress Testing: Baseline / Mild / Moderate / Severe scenarios."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.theme import (
    BAND_COLORS,
    configure_page,
    demo_data_disclaimer,
    require_artifacts,
    simulation_disclaimer,
)
from src.llm.tools.registry import call_tool

configure_page("Portfolio Stress Testing", icon="⚠️")
st.title("Portfolio Stress Testing")
demo_data_disclaimer()
simulation_disclaimer(
    "Stress scenarios are hypothetical simulations built by shocking income, expenses "
    "and utilization and re-scoring with the calibrated model. They are not forecasts."
)

if not require_artifacts():
    st.stop()

scenario = st.selectbox("Scenario", ["baseline", "mild", "moderate", "severe"], index=3)
result = call_tool("run_stress_test", {"scenario": scenario})

if not result["ok"]:
    st.error(result["error"])
    st.stop()

data = result["data"]
baseline, stressed = data["baseline"], data["stressed"]

st.markdown(f"#### Shock parameters: `{scenario}`")
st.json(data["shock_parameters"])

c1, c2, c3 = st.columns(3)
c1.metric(
    "Average PD",
    f"{stressed['average_pd']:.2%}",
    f"{(stressed['average_pd'] - baseline['average_pd']) * 100:+.2f} pp",
)
c2.metric(
    "Expected Loss",
    f"${stressed['total_expected_loss']:,.0f}",
    f"{stressed['total_expected_loss'] - baseline['total_expected_loss']:+,.0f}",
)
c3.metric(
    "Expected Loss Rate",
    f"{stressed['expected_loss_rate']:.2%}",
    f"{(stressed['expected_loss_rate'] - baseline['expected_loss_rate']) * 100:+.2f} pp",
)

st.markdown("---")
st.subheader("Risk Band Distribution: Baseline vs Stressed")
bands = list("ABCDE")
fig = go.Figure()
fig.add_trace(
    go.Bar(
        name="Baseline",
        x=bands,
        y=[baseline["band_distribution"].get(b, 0) for b in bands],
        marker_color="#94A3B8",
    )
)
fig.add_trace(
    go.Bar(
        name="Stressed",
        x=bands,
        y=[stressed["band_distribution"].get(b, 0) for b in bands],
        marker_color=[BAND_COLORS[b] for b in bands],
    )
)
fig.update_layout(barmode="group", height=420)
st.plotly_chart(fig, width="stretch")

st.markdown("---")
st.subheader("All Scenarios Compared")
all_results = {}
for name in ["baseline", "mild", "moderate", "severe"]:
    r = call_tool("run_stress_test", {"scenario": name})
    if r["ok"]:
        all_results[name] = r["data"]["stressed"]["average_pd"]

fig2 = go.Figure(
    go.Bar(
        x=list(all_results.keys()),
        y=list(all_results.values()),
        marker_color=["#94A3B8", "#D6A419", "#D9732B", "#B3261E"],
    )
)
fig2.update_layout(height=350, yaxis_tickformat=".1%", yaxis_title="Average PD")
st.plotly_chart(fig2, width="stretch")
