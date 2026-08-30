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

configure_page("Risk Segmentation", icon="\U0001f3af")
st.title("Risk Segmentation (Bands A-E)")
demo_data_disclaimer()
st.caption(
    "Risk bands are an illustrative 300-900 score converted from PD via a standard "
    "points-to-double-odds scorecard formula. They are demonstrative only, not a "
    "real commercial policy - see MODEL_CARD.md."
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
        labels={"observed_default_rate": "Observed Default Rate", "x": "Band"},
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
        labels={"expected_loss": "Expected Loss", "x": "Band"},
    )
    fig2.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig2, width="stretch")

st.markdown("---")
st.subheader("Compare Two Bands")
bands = [b for b in "ABCDE" if b in merged["risk_band"].unique()]
c1, c2 = st.columns(2)
band_a = c1.selectbox("Band A", bands, index=0)
band_b = c2.selectbox("Band B", bands, index=min(3, len(bands) - 1))

if band_a != band_b:
    result = compare_segments(scored, features, "risk_band", band_a, band_b)
    left, right = st.columns(2)
    for col, band in [(left, band_a), (right, band_b)]:
        with col:
            st.markdown(f"#### Band {band}")
            data = result[band]
            st.metric("Customers", f"{data['n_customers']:,}")
            st.metric("Average PD", f"{data['average_pd']:.2%}")
            st.metric("Expected Loss", f"${data['expected_loss']:,.0f}")
            st.metric("Average DTI", f"{data['average_dti']:.2f}")
            st.metric("Average Utilization", f"{data['average_utilization']:.2%}")
            st.metric("Observed Default Rate", f"{data['observed_default_rate']:.2%}")
else:
    st.info("Select two different bands to compare.")
