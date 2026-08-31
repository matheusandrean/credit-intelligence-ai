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

configure_page("Teste de Estresse da Carteira", icon="⚠️")
st.title("Teste de Estresse da Carteira")
demo_data_disclaimer()
simulation_disclaimer(
    "Os cenários de estresse são simulações hipotéticas construídas ao chocar renda, "
    "despesas e utilização e reclassificar com o modelo calibrado. Não são previsões."
)

if not require_artifacts():
    st.stop()

SCENARIO_LABELS = {
    "baseline": "Base",
    "mild": "Leve",
    "moderate": "Moderado",
    "severe": "Severo",
}
scenario = st.selectbox(
    "Cenário",
    ["baseline", "mild", "moderate", "severe"],
    index=3,
    format_func=lambda s: SCENARIO_LABELS.get(s, s),
)
result = call_tool("run_stress_test", {"scenario": scenario})

if not result["ok"]:
    st.error(result["error"])
    st.stop()

data = result["data"]
baseline, stressed = data["baseline"], data["stressed"]

st.markdown(f"#### Parâmetros de choque: `{SCENARIO_LABELS.get(scenario, scenario)}`")
st.json(data["shock_parameters"])

c1, c2, c3 = st.columns(3)
c1.metric(
    "PD Média",
    f"{stressed['average_pd']:.2%}",
    f"{(stressed['average_pd'] - baseline['average_pd']) * 100:+.2f} p.p.",
)
c2.metric(
    "Expected Loss",
    f"${stressed['total_expected_loss']:,.0f}",
    f"{stressed['total_expected_loss'] - baseline['total_expected_loss']:+,.0f}",
)
c3.metric(
    "Taxa de Expected Loss",
    f"{stressed['expected_loss_rate']:.2%}",
    f"{(stressed['expected_loss_rate'] - baseline['expected_loss_rate']) * 100:+.2f} p.p.",
)

st.markdown("---")
st.subheader("Distribuição por Faixa de Risco: Base vs. Estressado")
bands = list("ABCDE")
fig = go.Figure()
fig.add_trace(
    go.Bar(
        name="Base",
        x=bands,
        y=[baseline["band_distribution"].get(b, 0) for b in bands],
        marker_color="#94A3B8",
    )
)
fig.add_trace(
    go.Bar(
        name="Estressado",
        x=bands,
        y=[stressed["band_distribution"].get(b, 0) for b in bands],
        marker_color=[BAND_COLORS[b] for b in bands],
    )
)
fig.update_layout(barmode="group", height=420)
st.plotly_chart(fig, width="stretch")

st.markdown("---")
st.subheader("Comparação de Todos os Cenários")
all_results = {}
for name in ["baseline", "mild", "moderate", "severe"]:
    r = call_tool("run_stress_test", {"scenario": name})
    if r["ok"]:
        all_results[name] = r["data"]["stressed"]["average_pd"]

fig2 = go.Figure(
    go.Bar(
        x=[SCENARIO_LABELS.get(k, k) for k in all_results],
        y=list(all_results.values()),
        marker_color=["#94A3B8", "#D6A419", "#D9732B", "#B3261E"],
    )
)
fig2.update_layout(height=350, yaxis_tickformat=".1%", yaxis_title="PD Média")
st.plotly_chart(fig2, width="stretch")
