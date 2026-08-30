"""Credit Portfolio: composition of the book across key dimensions."""

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

configure_page("Carteira de Crédito", icon="\U0001f4bc")
st.title("Carteira de Crédito")
demo_data_disclaimer()

if not require_artifacts():
    st.stop()

scored = load_scored_portfolio()
features = load_features()
merged = scored.merge(features, on="customer_id", how="left", suffixes=("", "_feat"))

col1, col2 = st.columns(2)
with col1:
    st.subheader("Exposição por Faixa de Risco")
    exposure_by_band = merged.groupby("risk_band")["ead"].sum().reindex(list("ABCDE")).dropna()
    fig = px.bar(
        x=exposure_by_band.index,
        y=exposure_by_band.values,
        labels={"x": "Faixa de Risco", "y": "Exposição (EAD)"},
        color=exposure_by_band.index,
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, width="stretch")

with col2:
    st.subheader("Clientes por Faixa Etária")
    age_counts = merged["age_band"].value_counts().sort_index()
    fig2 = px.bar(
        x=age_counts.index, y=age_counts.values, labels={"x": "Faixa Etária", "y": "Clientes"}
    )
    fig2.update_traces(marker_color="#0B3D91")
    fig2.update_layout(height=380)
    st.plotly_chart(fig2, width="stretch")

col3, col4 = st.columns(2)
with col3:
    st.subheader("Distribuição de Renda")
    fig3 = px.histogram(merged, x="monthly_income", nbins=50)
    fig3.update_traces(marker_color="#0B3D91")
    fig3.update_layout(height=380, xaxis_title="Renda mensal")
    st.plotly_chart(fig3, width="stretch")

with col4:
    st.subheader("Distribuição de Comprometimento de Renda (DTI)")
    fig4 = px.histogram(merged, x="debt_to_income", nbins=50)
    fig4.update_traces(marker_color="#0B3D91")
    fig4.update_layout(height=380, xaxis_title="DTI")
    st.plotly_chart(fig4, width="stretch")

st.markdown("---")
st.subheader("Composição do Histórico de Pagamentos")
payment_counts = merged["payment_history"].value_counts()
fig5 = px.pie(
    names=payment_counts.index,
    values=payment_counts.values,
    color_discrete_sequence=px.colors.sequential.Blues_r,
)
fig5.update_layout(height=380)
st.plotly_chart(fig5, width="stretch")

st.markdown("---")
st.subheader("Amostra Bruta da Carteira")
st.dataframe(
    merged[
        [
            "customer_id",
            "age_band",
            "monthly_income",
            "debt_to_income",
            "credit_utilization",
            "risk_band",
            "pd",
            "score",
        ]
    ].head(200),
    width="stretch",
)
