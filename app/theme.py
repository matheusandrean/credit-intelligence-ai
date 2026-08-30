"""Shared Streamlit page configuration, styling and cached data loaders.

Every page imports from here so the whole app looks and behaves like one
coherent product (banking / fintech executive analytics), not a stack of
independently-styled demo scripts, and so heavy artifacts (parquet files,
the trained model, the SHAP explainer) are loaded from disk only once per
process instead of on every page rerun.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.llm.tools.context import get_tool_context
from src.utils.config import PROJECT_ROOT, get_settings

PRIMARY = "#0B3D91"
INK = "#101828"
MUTED = "#475467"
SURFACE = "#F5F7FA"
BAND_COLORS = {
    "A": "#1B7A4C",
    "B": "#4C9F70",
    "C": "#D6A419",
    "D": "#D9732B",
    "E": "#B3261E",
}

CUSTOM_CSS = f"""
<style>
    .stApp {{ background-color: {SURFACE}; }}
    h1, h2, h3 {{ color: {INK}; font-weight: 650; }}
    p, li, span, label {{ color: {INK}; }}
    [data-testid="stMetricValue"] {{ color: {PRIMARY}; font-weight: 700; }}
    [data-testid="stMetricLabel"] {{ color: {MUTED}; }}
    .disclaimer-box {{
        background-color: #FFF7E6;
        border-left: 4px solid #D6A419;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        font-size: 0.9rem;
        color: {INK};
        margin-bottom: 1rem;
    }}
    .sim-box {{
        background-color: #EAF1FB;
        border-left: 4px solid {PRIMARY};
        padding: 0.75rem 1rem;
        border-radius: 4px;
        font-size: 0.9rem;
        color: {INK};
        margin-bottom: 1rem;
    }}
</style>
"""


def configure_page(title: str, icon: str = "\U0001f4ca") -> None:
    st.set_page_config(
        page_title=f"{title} | Credit Intelligence AI", page_icon=icon, layout="wide"
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def demo_data_disclaimer() -> None:
    st.markdown(
        '<div class="disclaimer-box">'
        "This platform runs entirely on <strong>synthetic, demonstration data</strong>. "
        "No real customer, financial or personal information is used anywhere in this "
        "project. Risk bands, thresholds and policies shown are illustrative only."
        "</div>",
        unsafe_allow_html=True,
    )


def simulation_disclaimer(
    text: str = "This is a hypothetical simulation, not an observed outcome.",
) -> None:
    st.markdown(f'<div class="sim-box">{text}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_scored_portfolio() -> pd.DataFrame:
    settings = get_settings()
    return pd.read_parquet(settings.data_path / "processed" / "scored_portfolio.parquet")


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    settings = get_settings()
    return pd.read_parquet(settings.data_path / "processed" / "credit_features.parquet")


@st.cache_data(show_spinner=False)
def load_monthly_performance() -> pd.DataFrame:
    settings = get_settings()
    return pd.read_parquet(settings.data_path / "raw" / "monthly_performance.parquet")


@st.cache_data(show_spinner=False)
def load_customers_raw() -> pd.DataFrame:
    settings = get_settings()
    return pd.read_parquet(settings.data_path / "raw" / "customer_portfolio.parquet")


@st.cache_data(show_spinner=False)
def load_model_metadata() -> dict:
    settings = get_settings()
    return json.loads((settings.models_path / "model_metadata.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_json_report(filename: str) -> dict | None:
    path = PROJECT_ROOT / "reports" / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def get_cached_tool_context():
    return get_tool_context()


def artifacts_available() -> bool:
    settings = get_settings()
    return (settings.models_path / "model_metadata.json").exists() and (
        settings.data_path / "processed" / "scored_portfolio.parquet"
    ).exists()


def require_artifacts() -> bool:
    """Show a helpful message and return False if the pipeline hasn't been run yet."""
    if artifacts_available():
        return True
    st.warning(
        "No trained model artifacts were found yet. Run the pipeline first:\n\n"
        "```bash\nmake data\nmake features\nmake train\n```\n\n"
        "(or `.\\scripts\\tasks.ps1 data|features|train` on Windows)."
    )
    return False
