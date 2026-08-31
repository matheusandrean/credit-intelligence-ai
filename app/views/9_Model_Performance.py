"""Model Performance: discrimination, calibration and lift for the champion model."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.theme import (
    configure_page,
    demo_data_disclaimer,
    load_json_report,
    load_model_metadata,
    require_artifacts,
)

configure_page("Performance do Modelo", icon="\U0001f3c6")
st.title("Performance do Modelo")
demo_data_disclaimer()

if not require_artifacts():
    st.stop()

metadata = load_model_metadata()
comparison = load_json_report("model_comparison.json")
calibration = load_json_report("calibration_report.json")

st.markdown(
    f"**Modelo campeão:** `{metadata.get('champion_model')}` | "
    f"**Calibração:** `{metadata.get('calibration_method', 'n/a')}`"
)

if comparison:
    st.subheader("Campeão vs. Desafiante — Métricas do Teste Out-of-Time (OOT)")
    rows = []
    for model_name, splits in comparison.items():
        test_metrics = splits.get("test", {})
        rows.append(
            {
                "model": model_name,
                "roc_auc": test_metrics.get("roc_auc"),
                "ks_statistic": test_metrics.get("ks_statistic"),
                "gini": test_metrics.get("gini"),
                "brier_score": test_metrics.get("brier_score"),
                "lift_at_top_decile": test_metrics.get("lift_at_top_decile"),
                "recall_at_top_decile": test_metrics.get("recall_at_top_decile"),
            }
        )
    import pandas as pd

    df = pd.DataFrame(rows).set_index("model")
    df.index.name = "Modelo"
    df = df.rename(
        columns={
            "roc_auc": "ROC-AUC",
            "ks_statistic": "KS",
            "gini": "Gini",
            "brier_score": "Brier Score",
            "lift_at_top_decile": "Lift (decil superior)",
            "recall_at_top_decile": "Recall (decil superior)",
        }
    )
    st.dataframe(df.style.format("{:.4f}"), width="stretch")

    fig = go.Figure(go.Bar(x=df.index, y=df["ROC-AUC"], marker_color=["#0B3D91", "#94A3B8"]))
    fig.update_layout(title="ROC-AUC (teste OOT)", height=350, yaxis_range=[0.5, 1.0])
    st.plotly_chart(fig, width="stretch")
else:
    st.warning(
        "Nenhum relatório de comparação de modelos encontrado. Execute `make train` primeiro."
    )

st.markdown("---")
st.subheader("Calibração: Curva de Confiabilidade")
if calibration:
    c1, c2 = st.columns(2)
    c1.metric("Brier score bruto (teste)", f"{calibration['raw_brier_test']:.4f}")
    c2.metric("Brier score calibrado (teste)", f"{calibration['calibrated_brier_test']:.4f}")

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Calibração perfeita",
            line=dict(dash="dash", color="gray"),
        )
    )
    raw = calibration["reliability_raw_test"]
    cal = calibration["reliability_calibrated_test"]
    fig2.add_trace(
        go.Scatter(
            x=raw["mean_predicted"], y=raw["mean_observed"], mode="lines+markers", name="Bruto"
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=cal["mean_predicted"],
            y=cal["mean_observed"],
            mode="lines+markers",
            name="Calibrado",
        )
    )
    fig2.update_layout(
        height=450,
        xaxis_title="PD média prevista",
        yaxis_title="Taxa média de inadimplência observada",
    )
    st.plotly_chart(fig2, width="stretch")
else:
    st.warning(
        "Nenhum relatório de calibração encontrado. Execute `make train` e depois a "
        "etapa de calibração."
    )
