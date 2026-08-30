"""Governance: model, data and responsible-AI documentation, in one place."""

from __future__ import annotations

import streamlit as st

from app.theme import PROJECT_ROOT, configure_page, demo_data_disclaimer

configure_page("Governança", icon="\U0001f4dc")
st.title("Governança")
demo_data_disclaimer()

DOCS = {
    "Ficha do Modelo": "MODEL_CARD.md",
    "Ficha de Dados": "DATA_CARD.md",
    "Política de Governança": "GOVERNANCE.md",
    "IA Responsável": "RESPONSIBLE_AI.md",
}

tabs = st.tabs(list(DOCS.keys()))
for tab, filename in zip(tabs, DOCS.values(), strict=False):
    with tab:
        path = PROJECT_ROOT / filename
        if path.exists():
            st.markdown(path.read_text(encoding="utf-8"))
        else:
            st.info(f"`{filename}` ainda não foi gerado.")

st.markdown("---")
st.subheader("Base de Conhecimento de Políticas de Crédito Sintéticas")
st.caption(
    "Documentos usados pela geração aumentada por recuperação (RAG) do Analista de "
    "Crédito com IA."
)
kb_dir = PROJECT_ROOT / "knowledge_base"
if kb_dir.exists():
    for md_file in sorted(kb_dir.glob("*.md")):
        with st.expander(md_file.stem.replace("_", " ").title()):
            st.markdown(md_file.read_text(encoding="utf-8"))
