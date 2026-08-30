"""Governance: model, data and responsible-AI documentation, in one place."""

from __future__ import annotations

import streamlit as st

from app.theme import PROJECT_ROOT, configure_page, demo_data_disclaimer

configure_page("Governance", icon="\U0001f4dc")
st.title("Governance")
demo_data_disclaimer()

DOCS = {
    "Model Card": "MODEL_CARD.md",
    "Data Card": "DATA_CARD.md",
    "Governance Policy": "GOVERNANCE.md",
    "Responsible AI": "RESPONSIBLE_AI.md",
}

tabs = st.tabs(list(DOCS.keys()))
for tab, filename in zip(tabs, DOCS.values(), strict=False):
    with tab:
        path = PROJECT_ROOT / filename
        if path.exists():
            st.markdown(path.read_text(encoding="utf-8"))
        else:
            st.info(f"`{filename}` has not been generated yet.")

st.markdown("---")
st.subheader("Synthetic Credit Policy Knowledge Base")
st.caption("Documents used by the AI Credit Analyst's retrieval-augmented generation (RAG).")
kb_dir = PROJECT_ROOT / "knowledge_base"
if kb_dir.exists():
    for md_file in sorted(kb_dir.glob("*.md")):
        with st.expander(md_file.stem.replace("_", " ").title()):
            st.markdown(md_file.read_text(encoding="utf-8"))
