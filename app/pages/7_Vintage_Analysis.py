"""Vintage / Months-on-Book (MOB) analysis."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.theme import (
    configure_page,
    demo_data_disclaimer,
    load_customers_raw,
    load_monthly_performance,
    require_artifacts,
)
from src.analytics.vintage import build_vintage_curves, cohort_summary, filter_material_cohorts

configure_page("Vintage Analysis", icon="\U0001f4c5")
st.title("Vintage / MOB Analysis")
demo_data_disclaimer()
st.caption(
    "Bad-rate curves by origination cohort (vintage) and months-on-book (MOB), used to "
    "separate genuine credit-quality deterioration from portfolio growth or seasoning."
)

if not require_artifacts():
    st.stop()

panel = load_monthly_performance()
customers = load_customers_raw()

min_accounts = st.slider("Minimum accounts per cohort (filters noisy legacy cohorts)", 10, 200, 30)
curves = build_vintage_curves(panel, customers)
curves = filter_material_cohorts(curves, min_accounts=min_accounts)

recent_cohorts = sorted(curves["origination_cohort"].unique())[-12:]
plot_df = curves[curves["origination_cohort"].isin(recent_cohorts)]

fig = px.line(
    plot_df,
    x="mob",
    y="bad_rate",
    color="origination_cohort",
    markers=True,
    labels={
        "mob": "Months on Book",
        "bad_rate": "Bad Rate (61+ DPD)",
        "origination_cohort": "Cohort",
    },
)
fig.update_layout(height=480, yaxis_tickformat=".1%")
st.plotly_chart(fig, width="stretch")

st.markdown("---")
st.subheader("Cohort Ranking at a Fixed MOB Checkpoint")
checkpoint = st.slider(
    "MOB checkpoint",
    0,
    int(curves["mob"].max()) if not curves.empty else 0,
    min(6, int(curves["mob"].max()) if not curves.empty else 0),
)
summary = cohort_summary(curves, mob_checkpoint=checkpoint)
st.dataframe(summary, width="stretch")
