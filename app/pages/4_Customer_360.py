"""Customer 360: single-customer deep dive with SHAP explanation."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.theme import BAND_COLORS, configure_page, demo_data_disclaimer, require_artifacts
from src.llm.tools.registry import call_tool

configure_page("Customer 360", icon="\U0001f464")
st.title("Customer 360")
demo_data_disclaimer()

if not require_artifacts():
    st.stop()

default_id = "CUST_000001"
customer_id = st.text_input("Customer ID", value=default_id, help="e.g. CUST_000123")

if not customer_id:
    st.stop()

profile_result = call_tool("customer_risk_profile", {"customer_id": customer_id.strip().upper()})

if not profile_result["ok"]:
    st.error(profile_result["error"])
    st.stop()

data = profile_result["data"]
band = data["risk_band"]

st.markdown(f"### {customer_id.upper()} — Band **{band}**")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Probability of Default", f"{data['pd']:.2%}")
c2.metric("Illustrative Score", f"{data['score']:.0f}")
c3.metric("Risk Band", band)
c4.metric("Expected Loss", f"${data['expected_loss']:,.2f}")

behavior = data.get("behavior", {})
if behavior:
    st.markdown("#### Observed Behavior")
    b1, b2, b3 = st.columns(3)
    b1.metric(
        "Monthly income",
        f"${behavior['monthly_income']:,.0f}" if behavior["monthly_income"] else "n/a",
    )
    b2.metric(
        "Debt-to-Income",
        f"{behavior['debt_to_income']:.2f}" if behavior["debt_to_income"] else "n/a",
    )
    b3.metric(
        "Credit utilization",
        (
            f"{behavior['credit_utilization']:.2%}"
            if behavior["credit_utilization"] is not None
            else "n/a"
        ),
    )
    b4, b5, b6 = st.columns(3)
    b4.metric("Late payments (12m)", behavior["late_payments_12m"])
    b5.metric("Previous default", "Yes" if behavior["previous_default_flag"] else "No")
    b6.metric("Account tenure (months)", behavior["account_tenure_months"])

st.markdown("---")
st.subheader("Explainability (SHAP)")
st.caption(
    "Top factors are computed by SHAP on the champion model's raw decision function. "
    "The LLM layer is only allowed to narrate these values, never invent new ones."
)

shap_result = call_tool(
    "get_customer_shap", {"customer_id": customer_id.strip().upper(), "top_k": 5}
)
if shap_result["ok"]:
    shap_data = shap_result["data"]
    increasing = pd.DataFrame(shap_data["top_risk_increasing"])
    reducing = pd.DataFrame(shap_data["top_risk_reducing"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top factors increasing risk**")
        if not increasing.empty:
            fig = go.Figure(
                go.Bar(
                    x=increasing["shap_value"],
                    y=[f"{r.feature} = {r.value}" for r in increasing.itertuples()],
                    orientation="h",
                    marker_color=BAND_COLORS["E"],
                )
            )
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No risk-increasing factors identified.")
    with col2:
        st.markdown("**Top factors reducing risk**")
        if not reducing.empty:
            fig2 = go.Figure(
                go.Bar(
                    x=reducing["shap_value"],
                    y=[f"{r.feature} = {r.value}" for r in reducing.itertuples()],
                    orientation="h",
                    marker_color=BAND_COLORS["A"],
                )
            )
            fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("No risk-reducing factors identified.")
else:
    st.warning(shap_result["error"])
