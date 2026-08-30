"""AI Credit Analyst: chat interface over the Credit Intelligence Agent."""

from __future__ import annotations

import streamlit as st

from app.theme import configure_page, demo_data_disclaimer, require_artifacts
from src.agents.credit_intelligence_agent import CreditIntelligenceAgent
from src.utils.config import get_settings

configure_page("AI Credit Analyst", icon="\U0001f916")
st.title("AI Credit Analyst")
demo_data_disclaimer()

st.markdown("""
This assistant investigates the synthetic portfolio using typed tools (never
raw, unrestricted access). It **never approves or denies credit** and always
distinguishes observed data, model predictions, interpretation, and
hypothetical simulation. See `RESPONSIBLE_AI.md`.
""")

if not require_artifacts():
    st.stop()

settings = get_settings()
st.caption(f"Active LLM provider: **{settings.llm_provider}**")

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
DEMO_QUESTIONS_PT = [
    "Qual o perfil dos clientes de maior risco?",
    "Quais fatores mais contribuíram para o aumento da PD?",
    "Compare clientes das faixas A e D.",
    "Mostre a distribuição de risco por faixa de renda.",
    "O que aconteceria com a carteira se a renda dos clientes caísse 10%?",
    "Explique por que este cliente está classificado como risco elevado: CUST_000001.",
    "Quais políticas de crédito estão relacionadas a comprometimento elevado?",
    "Gere um resumo executivo da carteira.",
]

with st.expander("Example questions (English / Português)"):
    tab_en, tab_pt = st.tabs(["English", "Português"])
    with tab_en:
        for q in DEMO_QUESTIONS_EN:
            st.markdown(f"- {q}")
    with tab_pt:
        for q in DEMO_QUESTIONS_PT:
            st.markdown(f"- {q}")

if "agent" not in st.session_state:
    st.session_state.agent = CreditIntelligenceAgent()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + "; ".join(message["sources"]))
        if message.get("tools_called"):
            st.caption("Tools called: " + ", ".join(message["tools_called"]))

question = st.chat_input("Ask about the portfolio, a customer, a simulation, or a policy...")
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = st.session_state.agent.ask(question)
        st.markdown(response.answer)
        if response.sources:
            st.caption("Sources: " + "; ".join(response.sources))
        if response.tools_called:
            st.caption("Tools called: " + ", ".join(response.tools_called))

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
            "tools_called": response.tools_called,
        }
    )

if st.session_state.chat_history and st.button("Clear conversation"):
    st.session_state.chat_history = []
    st.rerun()
