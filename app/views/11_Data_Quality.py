"""Data Quality: automated validation report."""

from __future__ import annotations

import streamlit as st

from app.theme import configure_page, demo_data_disclaimer, load_json_report

configure_page("Qualidade dos Dados", icon="\U0001f9ea")
st.title("Qualidade dos Dados")
demo_data_disclaimer()

report = load_json_report("data_quality_report.json")
if not report:
    st.warning(
        "Nenhum relatório de qualidade de dados encontrado. Execute `make validate` primeiro."
    )
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Linhas validadas", f"{report['n_rows']:,}")
c2.metric("Schema válido", "Sim" if report["schema_valid"] else "Não")
c3.metric("Aprovado no geral", "Sim" if report["passed"] else "Não")

st.markdown("---")
st.subheader("IDs de Clientes Duplicados")
st.metric("Duplicatas encontradas", report["duplicate_customer_ids"])

st.subheader("Verificação de Atributos Protegidos")
if report["protected_attributes_found"]:
    st.error(f"Atributos protegidos encontrados: {report['protected_attributes_found']}")
else:
    st.success("Nenhum atributo protegido/sensível presente no conjunto de dados.")

st.subheader("Consistência entre Campos")
if report["consistency_failures"]:
    for f in report["consistency_failures"]:
        st.error(f)
else:
    st.success(
        "Nenhuma falha de consistência detectada (monotonicidade de atrasos, ordenação de datas)."
    )

st.markdown("---")
st.subheader("Taxa de Valores Nulos por Coluna")
null_rates = {k: v for k, v in report["null_rates"].items() if v > 0}
if null_rates:
    st.bar_chart(null_rates)
else:
    st.info("Nenhum valor ausente detectado.")

st.subheader("Anomalias de Volume Mensal (|z| > 2,5)")
if report["monthly_volume_anomalies"]:
    st.dataframe(report["monthly_volume_anomalies"], width="stretch")
else:
    st.success("Nenhuma anomalia de volume mensal detectada.")

st.markdown("---")
st.subheader("Taxa de Inadimplência por Mês")
st.line_chart(report["default_rate_by_month"])
