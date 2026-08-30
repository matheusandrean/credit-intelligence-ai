"""The Credit Intelligence Agent: orchestrates the LLM <-> tools loop.

Conceptual flow (see docs/ARCHITECTURE.md for the Mermaid diagram):

    User question -> LLM provider -> tool call(s) -> typed tool execution
    -> results back to LLM -> (repeat while more tools requested) -> final
    grounded answer -> audit log.

A lightweight custom loop was used instead of LangGraph for this project:
see docs/ARCHITECTURE.md ("Why not LangGraph") for the reasoning. The same
conceptual graph (intent -> tools -> evidence aggregation -> guardrail ->
response) is implemented directly here, just without the extra framework
dependency and its version-compatibility risk for a portfolio project that
needs to run reliably offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.llm.audit import ToolCallAudit, log_interaction
from src.llm.providers.base import BaseLLMProvider, ToolResultMessage
from src.llm.providers.factory import build_provider
from src.llm.tools.context import get_tool_context
from src.llm.tools.registry import call_tool
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

MAX_TOOL_ITERATIONS = 6


@dataclass
class AgentResponse:
    answer: str
    provider: str
    tools_called: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


class CreditIntelligenceAgent:
    """Stateless per-question wrapper around a stateful provider conversation.

    Each call to `ask` starts a fresh provider conversation (no cross-question
    memory), matching the platform's decision-support, non-persistent-session
    design; the dashboard's chat page keeps its own message history for
    display purposes only.
    """

    def __init__(self, provider: BaseLLMProvider | None = None):
        self.provider = provider or build_provider()

    def ask(self, question: str) -> AgentResponse:
        response = self.provider.send_message(question)
        tools_audit: list[ToolCallAudit] = []
        sources: list[str] = []
        iterations = 0

        while response.tool_calls and iterations < MAX_TOOL_ITERATIONS:
            results = []
            for call in response.tool_calls:
                result = call_tool(call.name, call.arguments)
                tools_audit.append(
                    ToolCallAudit(
                        name=call.name, ok=bool(result.get("ok", True)), error=result.get("error")
                    )
                )
                if call.name == "retrieve_credit_policy" and result.get("ok"):
                    for ev in result.get("data", {}).get("evidence", []):
                        sources.append(f"{ev['document']} > {ev['section']}")
                results.append(
                    ToolResultMessage(call_id=call.call_id, name=call.name, content=result)
                )
            response = self.provider.send_tool_results(results)
            iterations += 1

        answer = (
            response.text
            or "I was not able to produce a grounded answer to this question with the "
            "available tools. Please rephrase or ask about the portfolio, a specific "
            "customer, model metrics, a simulation, or credit policy."
        )

        model_metadata = get_tool_context().model_metadata
        model_version = (
            f"{model_metadata.get('champion_model')}"
            f"+{model_metadata.get('calibration_method', 'uncalibrated')}"
        )
        log_interaction(
            provider=self.provider.provider_name,
            question=question,
            tools_called=tools_audit,
            rag_sources=sources,
            model_version=model_version,
        )

        return AgentResponse(
            answer=answer,
            provider=self.provider.provider_name,
            tools_called=[t.name for t in tools_audit],
            sources=sorted(set(sources)),
        )


def get_default_agent() -> CreditIntelligenceAgent:
    settings = get_settings()
    logger.info("credit_intelligence_agent_ready", llm_provider=settings.llm_provider)
    return CreditIntelligenceAgent()
