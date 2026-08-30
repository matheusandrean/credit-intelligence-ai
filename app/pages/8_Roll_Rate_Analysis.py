"""Roll Rate Analysis: delinquency-bucket transition matrix."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.theme import (
    configure_page,
    demo_data_disclaimer,
    load_monthly_performance,
    require_artifacts,
)
from src.analytics.roll_rate import build_roll_rate_matrix, cure_and_migration_rates

configure_page("Roll Rate Analysis", icon="\U0001f504")
st.title("Roll Rate Analysis")
demo_data_disclaimer()
st.caption(
    "Month-over-month migration between delinquency buckets, pooled across the whole "
    "observation panel."
)

if not require_artifacts():
    st.stop()

panel = load_monthly_performance()
matrix = build_roll_rate_matrix(panel)

st.subheader("Transition Matrix")
fig = px.imshow(
    matrix.values,
    x=matrix.columns,
    y=matrix.index,
    color_continuous_scale="Blues",
    text_auto=".1%",
    labels={"x": "To bucket (next month)", "y": "From bucket (this month)", "color": "Probability"},
)
fig.update_layout(height=480)
st.plotly_chart(fig, width="stretch")

st.markdown("---")
st.subheader("Cure and Migration Rates")
rates = cure_and_migration_rates(matrix)
cols = st.columns(len(rates)) if rates else []
for col, (label, value) in zip(cols, rates.items(), strict=False):
    col.metric(label.replace("_", " ").title(), f"{value:.2%}")

st.markdown("---")
st.subheader("Raw Matrix")
st.dataframe(matrix.style.format("{:.2%}"), width="stretch")
