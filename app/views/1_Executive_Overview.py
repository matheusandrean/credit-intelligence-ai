"""Executive Overview: portfolio-level MIS KPIs with MoM movement."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.theme import (
    BAND_COLORS,
    configure_page,
    demo_data_disclaimer,
    load_features,
    load_scored_portfolio,
    require_artifacts,
)
from src.analytics.portfolio import monthly_kpi_trend, portfolio_summary

configure_page("Visão Executiva", icon="\U0001f4c8")
st.title("Visão Executiva")
demo_data_disclaimer()

if not require_artifacts():
    st.stop()

scored = load_scored_portfolio()
features = load_features()
summary = portfolio_summary(scored, features)
trend = monthly_kpi_trend(scored, features)

st.subheader("KPIs da Carteira")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de clientes", f"{summary['total_customers']:,}")
c2.metric("Exposição da carteira", f"${summary['portfolio_exposure']:,.0f}")
c3.metric("PD média", f"{summary['average_pd']:.2%}")
c4.metric("Expected Loss", f"${summary['expected_loss']:,.0f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Taxa de inadimplência observada", f"{summary['observed_default_rate']:.2%}")
c6.metric("Taxa de atraso (3m)", f"{summary['delinquency_rate_3m']:.2%}")
c7.metric("População de alto risco (D+E)", f"{summary['high_risk_population_pct']:.2%}")
c8.metric("Score médio", f"{summary['average_score']:.0f}")

st.markdown("---")
st.subheader("Tendência Mensal")

if len(trend) > 1:
    latest, prior = trend.iloc[-1], trend.iloc[-2]
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "PD média (var. mensal)",
        f"{latest['average_pd']:.2%}",
        f"{(latest['average_pd'] - prior['average_pd']) * 100:+.2f} p.p.",
    )
    m2.metric(
        "Expected Loss (var. mensal)",
        f"${latest['expected_loss']:,.0f}",
        f"{latest['expected_loss'] - prior['expected_loss']:+,.0f}",
    )
    m3.metric(
        "% de alto risco (var. mensal)",
        f"{latest['high_risk_population_pct']:.2%}",
        f"{(latest['high_risk_population_pct'] - prior['high_risk_population_pct']) * 100:+.2f} p.p.",
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend["month"], y=trend["average_pd"], name="PD média", line=dict(color="#0B3D91")
        )
    )
    fig.update_layout(
        title="PD média ao longo do tempo",
        yaxis_tickformat=".1%",
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            x=trend["month"],
            y=trend["expected_loss"],
            name="Expected Loss",
            marker_color="#0B3D91",
        )
    )
    fig2.update_layout(
        title="Expected Loss ao longo do tempo", height=380, margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig2, width="stretch")
else:
    st.info(
        "Não há histórico mensal suficiente neste conjunto de dados para calcular uma tendência."
    )

st.markdown("---")
st.subheader("Distribuição por Faixa de Risco")
band_dist = summary["risk_band_distribution"]
fig3 = go.Figure(
    data=[
        go.Bar(
            x=list(band_dist.keys()),
            y=list(band_dist.values()),
            marker_color=[BAND_COLORS.get(b, "#999") for b in band_dist],
        )
    ]
)
fig3.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig3, width="stretch")

st.markdown("---")
st.subheader("Relatório Executivo")
if st.button("Gerar Relatório Executivo da Carteira"):
    from src.reporting.executive_report import generate_executive_report

    with st.spinner("Gerando relatório..."):
        report = generate_executive_report()
    st.markdown(report)
