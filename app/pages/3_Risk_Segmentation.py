"""Risk Segmentation: bands A-E and their behavior, with segment comparison."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.theme import (
    BAND_COLORS,
    configure_page,
    demo_data_disclaimer,
    load_features,
    load_scored_portfolio,
    require_artifacts,
)
from src.analytics.portfolio import compare_segments

configure_page("Segmentação de Risco", icon="\U0001f3af")
st.title("Segmentação de Risco (Faixas A-E)")
demo_data_disclaimer()
st.caption(
    "As faixas de risco são um score ilustrativo de 300-900 convertido a partir da PD "
    "por meio de uma fórmula padrão de scorecard (points-to-double-odds). São apenas "
    "demonstrativas, não uma política comercial real - veja MODEL_CARD.md."
)

if not require_artifacts():
    st.stop()

scored = load_scored_portfolio()
features = load_features()
merged = scored.merge(features, on="customer_id", how="left", suffixes=("", "_feat"))

band_stats = (
    merged.groupby("risk_band")
    .agg(
        n_customers=("customer_id", "count"),
        average_pd=("pd", "mean"),
        observed_default_rate=("target_default_90d", "mean"),
        expected_loss=("expected_loss", "sum"),
        average_dti=("debt_to_income", "mean"),
        average_utilization=("credit_utilization", "mean"),
    )
    .reindex(list("ABCDE"))
    .dropna(how="all")
)
st.dataframe(band_stats.style.format("{:.4f}"), width="stretch")

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(
        band_stats,
        x=band_stats.index,
        y="observed_default_rate",
        color=band_stats.index,
        color_discrete_map=BAND_COLORS,
        labels={"observed_default_rate": "Taxa de Inadimplência Observada", "x": "Faixa"},
    )
    fig.update_layout(showlegend=False, height=380, yaxis_tickformat=".1%")
    st.plotly_chart(fig, width="stretch")

with col2:
    fig2 = px.bar(
        band_stats,
        x=band_stats.index,
        y="expected_loss",
        color=band_stats.index,
        color_discrete_map=BAND_COLORS,
        labels={"expected_loss": "Expected Loss", "x": "Faixa"},
    )
    fig2.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig2, width="stretch")

st.markdown("---")
st.subheader("Comparar Duas Faixas")
bands = [b for b in "ABCDE" if b in merged["risk_band"].unique()]
c1, c2 = st.columns(2)
band_a = c1.selectbox("Faixa A", bands, index=0)
band_b = c2.selectbox("Faixa B", bands, index=min(3, len(bands) - 1))

if band_a != band_b:
    result = compare_segments(scored, features, "risk_band", band_a, band_b)
    left, right = st.columns(2)
    for col, band in [(left, band_a), (right, band_b)]:
        with col:
            st.markdown(f"#### Faixa {band}")
            data = result[band]
            st.metric("Clientes", f"{data['n_customers']:,}")
            st.metric("PD Média", f"{data['average_pd']:.2%}")
            st.metric("Expected Loss", f"${data['expected_loss']:,.0f}")
            st.metric("DTI Média", f"{data['average_dti']:.2f}")
            st.metric("Utilização Média", f"{data['average_utilization']:.2%}")
            st.metric("Taxa de Inadimplência Observada", f"{data['observed_default_rate']:.2%}")
else:
    st.info("Selecione duas faixas diferentes para comparar.")
