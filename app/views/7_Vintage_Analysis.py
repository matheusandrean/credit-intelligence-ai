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

configure_page("Análise de Safras", icon="\U0001f4c5")
st.title("Análise de Safras (Vintage / MOB)")
demo_data_disclaimer()
st.caption(
    "Curvas de taxa de inadimplência (bad rate) por safra de originação (vintage) e "
    "meses na base (MOB), usadas para separar deterioração genuína da qualidade de "
    "crédito de crescimento ou maturação da carteira."
)

if not require_artifacts():
    st.stop()

panel = load_monthly_performance()
customers = load_customers_raw()

min_accounts = st.slider(
    "Mínimo de contas por safra (filtra safras antigas com ruído)", 10, 200, 30
)
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
        "mob": "Meses na Base (MOB)",
        "bad_rate": "Taxa de Inadimplência (61+ DPD)",
        "origination_cohort": "Safra",
    },
)
fig.update_layout(height=480, yaxis_tickformat=".1%")
st.plotly_chart(fig, width="stretch")

st.markdown("---")
st.subheader("Ranking de Safras em um Ponto Fixo de MOB")
checkpoint = st.slider(
    "Ponto de checagem (MOB)",
    0,
    int(curves["mob"].max()) if not curves.empty else 0,
    min(6, int(curves["mob"].max()) if not curves.empty else 0),
)
summary = cohort_summary(curves, mob_checkpoint=checkpoint)
st.dataframe(summary, width="stretch")
