"""Integration tests for the full Credit Intelligence Agent loop
(Demo Mode provider, so no API key is required)."""

from __future__ import annotations

import pytest

from src.agents.credit_intelligence_agent import CreditIntelligenceAgent
from src.llm.providers.demo_provider import DemoProvider
from src.llm.system_prompt import SYSTEM_PROMPT
from src.llm.tools.registry import list_tool_specs
from src.utils.config import get_settings


def _artifacts_available() -> bool:
    settings = get_settings()
    return (settings.models_path / "champion_calibrated.joblib").exists() and (
        settings.data_path / "processed" / "scored_portfolio.parquet"
    ).exists()


pytestmark = pytest.mark.skipif(
    not _artifacts_available(), reason="trained model artifacts not present"
)


def _demo_agent() -> CreditIntelligenceAgent:
    provider = DemoProvider(SYSTEM_PROMPT, list_tool_specs())
    return CreditIntelligenceAgent(provider=provider)


def test_agent_answers_portfolio_question_with_real_numbers() -> None:
    agent = _demo_agent()
    response = agent.ask("What is the current risk profile of the portfolio?")
    assert response.tools_called == ["portfolio_summary"]
    assert "portfolio_summary" in response.answer


def test_agent_never_answers_approval_decision_by_fabricating_one() -> None:
    """The system prompt forbids autonomous decisions; the Demo provider,
    lacking free-form reasoning, at minimum must never silently succeed
    with an empty tool-less answer that could look like a decision."""
    agent = _demo_agent()
    response = agent.ask("Should CUST_000001 be approved for a credit increase?")
    assert response.tools_called  # it must have looked something up, not just replied from nothing


def test_agent_cites_policy_sources_for_policy_questions() -> None:
    agent = _demo_agent()
    response = agent.ask("Which policy addresses customers with high utilization?")
    assert response.tools_called == ["retrieve_credit_policy"]
    assert len(response.sources) > 0
    assert all(">" in s for s in response.sources)  # "Document > Section" format


def test_agent_logs_audit_record(tmp_path) -> None:
    import src.llm.audit as audit_module

    log_path = tmp_path / "audit_log.jsonl"
    original_path = audit_module.AUDIT_LOG_PATH
    audit_module.AUDIT_LOG_PATH = log_path
    try:
        agent = _demo_agent()
        agent.ask("portfolio summary please")
        records = audit_module.read_recent_audit_records(log_path=log_path)
        assert len(records) == 1
        assert records[0]["provider"] == "demo"
        assert records[0]["tools_called"][0]["name"] == "portfolio_summary"
    finally:
        audit_module.AUDIT_LOG_PATH = original_path


def test_agent_stress_test_response_is_grounded_and_labeled_hypothetical() -> None:
    agent = _demo_agent()
    response = agent.ask("What happens under the severe stress scenario?")
    assert response.tools_called == ["run_stress_test"]
    assert "is_simulation" in response.answer
