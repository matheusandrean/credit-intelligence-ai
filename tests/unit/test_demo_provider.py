"""Unit tests for the deterministic Demo Mode provider's intent routing."""

from __future__ import annotations

from src.llm.providers.demo_provider import DemoProvider
from src.llm.tools.registry import list_tool_specs

TOOLS = list_tool_specs()


def _classify(question: str) -> str:
    provider = DemoProvider("system", TOOLS)
    response = provider.send_message(question)
    assert response.tool_calls
    return response.tool_calls[0].name


def test_routes_portfolio_overview() -> None:
    assert _classify("What is the current risk profile of the portfolio?") == "portfolio_summary"


def test_routes_feature_importance() -> None:
    assert _classify("Which factors are driving default risk?") == "get_feature_importance"


def test_routes_compare_segments_with_bands() -> None:
    provider = DemoProvider("system", TOOLS)
    response = provider.send_message("Compare risk bands A and D.")
    call = response.tool_calls[0]
    assert call.name == "compare_segments"
    assert call.arguments["segment_a"] == "A"
    assert call.arguments["segment_b"] == "D"


def test_routes_customer_shap_when_explain_keyword_present() -> None:
    assert _classify("Explain why CUST_000123 has high risk.") == "get_customer_shap"


def test_routes_customer_profile_without_explain_keyword() -> None:
    assert _classify("Show me CUST_000123.") == "customer_risk_profile"


def test_routes_stress_test_with_severe_scenario() -> None:
    provider = DemoProvider("system", TOOLS)
    response = provider.send_message("What happens under the severe stress scenario?")
    call = response.tool_calls[0]
    assert call.name == "run_stress_test"
    assert call.arguments["scenario"] == "severe"


def test_routes_what_if_with_customer_and_shock() -> None:
    provider = DemoProvider("system", TOOLS)
    response = provider.send_message("Simulate what happens to CUST_000123 if income drops -10%.")
    call = response.tool_calls[0]
    assert call.name == "run_what_if"
    assert call.arguments["customer_id"] == "CUST_000123"
    assert call.arguments["income_shock_pct"] == -0.10


def test_routes_drift_detection() -> None:
    assert _classify("What are the main model drift indicators?") == "detect_drift"


def test_routes_policy_retrieval_english() -> None:
    assert _classify("Which policy addresses high utilization?") == "retrieve_credit_policy"


def test_routes_policy_retrieval_portuguese_no_accent() -> None:
    assert (
        _classify("Qual politica trata clientes com comprometimento elevado?")
        == "retrieve_credit_policy"
    )


def test_routes_model_metrics() -> None:
    assert _classify("What is the model's AUC and KS?") == "get_model_metrics"


def test_routes_executive_report() -> None:
    assert (
        _classify("Summarize the portfolio for a Credit Risk Director.")
        == "generate_executive_report"
    )
    assert _classify("Gere um resumo executivo da carteira.") == "generate_executive_report"


def test_demo_disclaimer_present_in_final_answer() -> None:
    from src.llm.providers.base import ToolResultMessage
    from src.llm.providers.demo_provider import DEMO_DISCLAIMER

    provider = DemoProvider("system", TOOLS)
    provider.send_message("portfolio overview")
    response = provider.send_tool_results(
        [ToolResultMessage(call_id="1", name="portfolio_summary", content={"ok": True, "data": {}})]
    )
    assert DEMO_DISCLAIMER in response.text
    assert response.tool_calls == []
