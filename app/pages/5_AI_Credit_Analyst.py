"""AI Credit Analyst: chat interface over the Credit Intelligence Agent."""

from __future__ import annotations

import streamlit as st

from app.theme import configure_page, demo_data_disclaimer, require_artifacts
from src.agents.credit_intelligence_agent import CreditIntelligenceAgent
from src.utils.config import get_settings

configure_page("Analista de Crédito com IA", icon="\U0001f916")
st.title("Analista de Crédito com IA")
demo_data_disclaimer()

st.markdown("""
Este assistente investiga a carteira sintética usando ferramentas tipadas (nunca
acesso bruto e irrestrito). Ele **nunca aprova ou nega crédito** e sempre distingue
dados observados, previsões do modelo, interpretação e simulação hipotética. Veja
`RESPONSIBLE_AI.md`.
""")

if not require_artifacts():
    st.stop()

settings = get_settings()
st.caption(f"Provedor de LLM ativo: **{settings.llm_provider}**")

DEMO_QUESTIONS_PT = [
    "Qual o perfil de risco atual da carteira?",
    "Quais fatores mais contribuíram para o aumento da PD?",
    "Compare as faixas de risco A e D.",
    "Quais segmentos mais deterioraram?",
    "O que mostra a última análise de safras (vintage)?",
    "Explique o perfil de risco do cliente CUST_000001.",
    "O que acontece no cenário de estresse severo?",
    "Quais são os principais indicadores de drift do modelo?",
    "Resuma a carteira para um Diretor de Risco de Crédito.",
]
DEMO_QUESTIONS_EN = [
    "What is the current risk profile of the portfolio?",
    "Which factors are driving default risk?",
    "Compare risk bands A and D.",
    "Which segments deteriorated the most?",
    "What does the latest vintage analysis show?",
    "Explain customer CUST_000001's risk profile.",
    "What happens under the severe stress scenario?",
    "What are the main model drift indicators?",
    "Summarize the portfolio for a Credit Risk Director.",
]

with st.expander("Perguntas de exemplo (Português / English)"):
    tab_pt, tab_en = st.tabs(["Português", "English"])
    with tab_pt:
        for q in DEMO_QUESTIONS_PT:
            st.markdown(f"- {q}")
    with tab_en:
        for q in DEMO_QUESTIONS_EN:
            st.markdown(f"- {q}")

if "agent" not in st.session_state:
    st.session_state.agent = CreditIntelligenceAgent()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Fontes: " + "; ".join(message["sources"]))
        if message.get("tools_called"):
            st.caption("Ferramentas utilizadas: " + ", ".join(message["tools_called"]))

question = st.chat_input("Pergunte sobre a carteira, um cliente, uma simulação ou uma política...")
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analisando..."):
            response = st.session_state.agent.ask(question)
        st.markdown(response.answer)
        if response.sources:
            st.caption("Fontes: " + "; ".join(response.sources))
        if response.tools_called:
            st.caption("Ferramentas utilizadas: " + ", ".join(response.tools_called))

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
            "tools_called": response.tools_called,
        }
    )

if st.session_state.chat_history and st.button("Limpar conversa"):
    st.session_state.chat_history = []
    st.rerun()
