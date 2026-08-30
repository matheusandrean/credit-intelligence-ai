"""Credit Intelligence AI - Streamlit application entry point."""

from __future__ import annotations

import streamlit as st

from app.theme import configure_page, demo_data_disclaimer, load_model_metadata, require_artifacts
from src.utils.config import get_settings


def home_page() -> None:
    configure_page("Início", icon="\U0001f3e6")

    st.title("Credit Intelligence AI")
    st.caption(
        "Uma plataforma híbrida de apoio à decisão que combina Risco de Crédito, IA "
        "Explicável e IA Generativa, para analistas de crédito, gestores de risco e "
        "equipes de MIS."
    )

    demo_data_disclaimer()

    st.markdown("""
### O que esta plataforma faz

O Credit Intelligence AI combina um **modelo quantitativo de risco de crédito**
(Regressão Logística + LightGBM, calibrado, explicado com SHAP) com uma **camada
de IA Generativa** que permite aos analistas investigar a carteira em linguagem
natural - sem que o LLM jamais aprove, negue ou substitua o modelo quantitativo.

Use a navegação à esquerda para explorar:

| Página | O que você vai encontrar |
|---|---|
| Visão Executiva | KPIs em nível de carteira, período atual vs. anterior |
| Carteira de Crédito | Composição da carteira pelas principais dimensões |
| Segmentação de Risco | Faixas de risco A-E e seu comportamento |
| Visão 360 do Cliente | Análise detalhada de um único cliente: PD, score, faixa, SHAP |
| Analista de Crédito com IA | Converse com o agente - pergunte sobre risco, causas, políticas |
| Teste de Estresse da Carteira | Cenários Base / Leve / Moderado / Severo |
| Análise de Safras | Curvas de inadimplência por safra de originação e MOB |
| Análise de Roll Rate | Matriz de migração entre faixas de atraso |
| Performance do Modelo | ROC-AUC, KS, Gini, calibração, lift |
| Monitoramento do Modelo | PSI de drift para variáveis e PD prevista ao longo do tempo |
| Qualidade dos Dados | Relatório automatizado de validação |
| Governança | Documentação de modelo, dados e IA responsável |
""")

    if require_artifacts():
        metadata = load_model_metadata()
        settings = get_settings()
        col1, col2, col3 = st.columns(3)
        col1.metric("Modelo campeão", metadata.get("champion_model", "n/a"))
        col2.metric("Calibração", metadata.get("calibration_method", "n/a"))
        col3.metric("Provedor de LLM (esta sessão)", settings.llm_provider)

    st.markdown("---")
    st.markdown("""
### Humano no controle, por design

Nenhum componente desta plataforma - modelo quantitativo ou agente de IA
generativa - está autorizado a aprovar, negar ou finalizar uma decisão de
crédito. Toda saída é um **apoio** à decisão de um analista humano. Veja
`RESPONSIBLE_AI.md` para a política completa.

**Aviso:** este é um projeto de portfólio/educacional, construído inteiramente
com dados sintéticos. Não deve ser usado para decisões de crédito reais sem
validação independente, governança, compliance e revisão jurídica.
""")

    st.markdown("---")
    st.caption("Credit Intelligence AI - desenvolvido por Matheus Marcondes.")


PAGES = [
    st.Page(home_page, title="Início", icon="\U0001f3e6", default=True, url_path="Home"),
    st.Page(
        "pages/1_Executive_Overview.py",
        title="Visão Executiva",
        icon="\U0001f4c8",
        url_path="Executive_Overview",
    ),
    st.Page(
        "pages/2_Credit_Portfolio.py",
        title="Carteira de Crédito",
        icon="\U0001f4bc",
        url_path="Credit_Portfolio",
    ),
    st.Page(
        "pages/3_Risk_Segmentation.py",
        title="Segmentação de Risco",
        icon="\U0001f3af",
        url_path="Risk_Segmentation",
    ),
    st.Page(
        "pages/4_Customer_360.py",
        title="Visão 360 do Cliente",
        icon="\U0001f464",
        url_path="Customer_360",
    ),
    st.Page(
        "pages/5_AI_Credit_Analyst.py",
        title="Analista de Crédito com IA",
        icon="\U0001f916",
        url_path="AI_Credit_Analyst",
    ),
    st.Page(
        "pages/6_Portfolio_Stress_Testing.py",
        title="Teste de Estresse da Carteira",
        icon="⚠️",
        url_path="Portfolio_Stress_Testing",
    ),
    st.Page(
        "pages/7_Vintage_Analysis.py",
        title="Análise de Safras",
        icon="\U0001f4c5",
        url_path="Vintage_Analysis",
    ),
    st.Page(
        "pages/8_Roll_Rate_Analysis.py",
        title="Análise de Roll Rate",
        icon="\U0001f504",
        url_path="Roll_Rate_Analysis",
    ),
    st.Page(
        "pages/9_Model_Performance.py",
        title="Performance do Modelo",
        icon="\U0001f3c6",
        url_path="Model_Performance",
    ),
    st.Page(
        "pages/10_Model_Monitoring.py",
        title="Monitoramento do Modelo",
        icon="\U0001f4e1",
        url_path="Model_Monitoring",
    ),
    st.Page(
        "pages/11_Data_Quality.py",
        title="Qualidade dos Dados",
        icon="\U0001f9ea",
        url_path="Data_Quality",
    ),
    st.Page(
        "pages/12_Governance.py",
        title="Governança",
        icon="\U0001f4dc",
        url_path="Governance",
    ),
]

st.navigation(PAGES).run()
