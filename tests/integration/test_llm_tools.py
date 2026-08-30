"""Integration tests for the LLM tool registry against real trained artifacts.

These exercise the exact call path the agent uses (`call_tool`), including
Pydantic validation and the SQL guardrail, so they double as regression
tests for the "no arbitrary code / SQL" safety requirement.
"""

from __future__ import annotations

import pytest

from src.llm.tools.registry import call_tool, list_tool_specs
from src.utils.config import get_settings


def _artifacts_available() -> bool:
    settings = get_settings()
    return (settings.models_path / "champion_calibrated.joblib").exists() and (
        settings.data_path / "processed" / "scored_portfolio.parquet"
    ).exists()


pytestmark = pytest.mark.skipif(
    not _artifacts_available(), reason="trained model artifacts not present"
)


def test_all_tools_are_registered() -> None:
    names = {spec.name for spec in list_tool_specs()}
    expected = {
        "portfolio_summary",
        "customer_risk_profile",
        "compare_segments",
        "calculate_default_rate",
        "calculate_expected_loss",
        "get_feature_importance",
        "get_customer_shap",
        "run_stress_test",
        "run_what_if",
        "query_portfolio",
        "retrieve_credit_policy",
        "detect_drift",
        "get_model_metrics",
    }
    assert expected <= names


def test_portfolio_summary_returns_ok() -> None:
    result = call_tool("portfolio_summary", {})
    assert result["ok"] is True
    assert result["data"]["total_customers"] > 0


def test_unknown_customer_returns_structured_error() -> None:
    result = call_tool("customer_risk_profile", {"customer_id": "CUST_999999"})
    assert result["ok"] is False
    assert "not found" in result["error"]


def test_invalid_arguments_return_structured_error() -> None:
    result = call_tool("get_feature_importance", {"top_n": -5})
    assert result["ok"] is False
    assert "Invalid arguments" in result["error"]


def test_unknown_tool_name_returns_structured_error() -> None:
    result = call_tool("delete_everything", {})
    assert result["ok"] is False
    assert "Unknown tool" in result["error"]


@pytest.mark.parametrize(
    "malicious_sql",
    [
        "DROP TABLE portfolio",
        "SELECT * FROM portfolio; DELETE FROM portfolio",
        "UPDATE portfolio SET pd = 0",
        "SELECT * FROM portfolio; -- comment\nDROP TABLE portfolio",
    ],
)
def test_query_portfolio_rejects_destructive_sql(malicious_sql: str) -> None:
    result = call_tool("query_portfolio", {"sql": malicious_sql})
    assert result["ok"] is False


def test_query_portfolio_allows_safe_select() -> None:
    result = call_tool("query_portfolio", {"sql": "SELECT COUNT(*) as n FROM portfolio"})
    assert result["ok"] is True
    assert result["data"]["rows"][0]["n"] > 0


def test_run_stress_test_reports_simulation_flag() -> None:
    result = call_tool("run_stress_test", {"scenario": "severe"})
    assert result["ok"] is True
    assert result["data"]["is_simulation"] is True
    assert result["data"]["stressed"]["average_pd"] >= result["data"]["baseline"]["average_pd"]


def test_retrieve_credit_policy_returns_citations() -> None:
    result = call_tool(
        "retrieve_credit_policy", {"question": "human oversight of credit decisions"}
    )
    assert result["ok"] is True
    assert len(result["data"]["evidence"]) > 0
    assert all("document" in e and "section" in e for e in result["data"]["evidence"])
