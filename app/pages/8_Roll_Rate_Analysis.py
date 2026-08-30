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

configure_page("Análise de Roll Rate", icon="\U0001f504")
st.title("Análise de Roll Rate")
demo_data_disclaimer()
st.caption("Migração mês a mês entre faixas de atraso, agregada em todo o painel de observação.")

if not require_artifacts():
    st.stop()

panel = load_monthly_performance()
matrix = build_roll_rate_matrix(panel)

st.subheader("Matriz de Transição")
fig = px.imshow(
    matrix.values,
    x=matrix.columns,
    y=matrix.index,
    color_continuous_scale="Blues",
    text_auto=".1%",
    labels={
        "x": "Faixa de destino (mês seguinte)",
        "y": "Faixa de origem (mês atual)",
        "color": "Probabilidade",
    },
)
fig.update_layout(height=480)
st.plotly_chart(fig, width="stretch")


def _translate_rate_label(key: str) -> str:
    if key == "roll_forward_rate_from_current":
        return "Taxa de avanço a partir de Em dia"
    if key.startswith("cure_rate_from_"):
        bucket = key.removeprefix("cure_rate_from_")
        return f"Taxa de cura a partir de {bucket}"
    return key.replace("_", " ").title()


st.markdown("---")
st.subheader("Taxas de Cura e Migração")
rates = cure_and_migration_rates(matrix)
cols = st.columns(len(rates)) if rates else []
for col, (label, value) in zip(cols, rates.items(), strict=False):
    col.metric(_translate_rate_label(label), f"{value:.2%}")

st.markdown("---")
st.subheader("Matriz Bruta")
st.dataframe(matrix.style.format("{:.2%}"), width="stretch")
