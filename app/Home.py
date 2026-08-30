"""Credit Intelligence AI - Streamlit application entry point."""

from __future__ import annotations

import streamlit as st

from app.theme import configure_page, demo_data_disclaimer, load_model_metadata, require_artifacts
from src.utils.config import get_settings

configure_page("Home", icon="\U0001f3e6")

st.title("Credit Intelligence AI")
st.caption(
    "A hybrid Credit Risk + Explainable AI + Generative AI decision-support platform "
    "for credit analysts, risk managers and MIS teams."
)

demo_data_disclaimer()

st.markdown("""
### What this platform does

Credit Intelligence AI combines a **quantitative credit risk model** (Logistic
Regression + LightGBM, calibrated, explained with SHAP) with a **Generative AI
layer** that lets analysts investigate the portfolio in natural language -
without the LLM ever approving, denying, or replacing the quantitative model.

Use the navigation on the left to explore:

| Page | What you'll find |
|---|---|
| Executive Overview | Portfolio-level KPIs, current vs. prior period |
| Credit Portfolio | Composition of the book by key dimensions |
| Risk Segmentation | Risk bands A-E and their behavior |
| Customer 360 | Single-customer deep dive: PD, score, band, SHAP |
| AI Credit Analyst | Chat with the agent - ask about risk, drivers, policy |
| Portfolio Stress Testing | Baseline / Mild / Moderate / Severe scenarios |
| Vintage Analysis | Bad-rate curves by origination cohort and MOB |
| Roll Rate Analysis | Delinquency-bucket migration matrix |
| Model Performance | ROC-AUC, KS, Gini, calibration, lift |
| Model Monitoring | PSI drift for features and predicted PD over time |
| Data Quality | Automated validation report |
| Governance | Model, data and responsible-AI documentation |
""")

if require_artifacts():
    metadata = load_model_metadata()
    settings = get_settings()
    col1, col2, col3 = st.columns(3)
    col1.metric("Champion model", metadata.get("champion_model", "n/a"))
    col2.metric("Calibration", metadata.get("calibration_method", "n/a"))
    col3.metric("LLM provider (this session)", settings.llm_provider)

st.markdown("---")
st.markdown("""
### Human-in-the-loop, by design

No component of this platform - quantitative model or generative-AI agent -
is authorized to approve, deny, or finalize a credit decision. Every output
is decision **support** for a human analyst. See `RESPONSIBLE_AI.md` for the
full policy.

**Disclaimer:** this is a portfolio/educational project built entirely on
synthetic data. It must not be used for real credit decisions without
independent validation, governance, compliance and legal review.
""")

st.markdown("---")
st.caption("Credit Intelligence AI - built by Matheus Marcondes.")
