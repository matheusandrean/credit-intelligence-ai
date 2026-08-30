"""Provider-agnostic LLM interface for the Credit Intelligence Agent.

Every provider (Anthropic, OpenAI, Ollama, Demo) implements the same
stateful conversation interface so the agent orchestrator
(`src/agents/credit_intelligence_agent.py`) never needs to know which
provider is active. Tool definitions are described once, provider-neutrally,
via `ToolSpec` (see `src/llm/tools/registry.py`) and each provider adapts
them to its own function-calling schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.llm.tools.registry import ToolSpec


@dataclass
class ToolCallRequest:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResultMessage:
    call_id: str
    name: str
    content: dict[str, Any]


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    stop_reason: str = "end_turn"
    raw_usage: dict[str, Any] | None = None


class BaseLLMProvider(ABC):
    """Stateful, single-conversation LLM provider adapter."""

    provider_name: str = "base"

    def __init__(self, system_prompt: str, tools: list[ToolSpec]):
        self.system_prompt = system_prompt
        self.tools = tools

    @abstractmethod
    def send_message(self, user_text: str) -> LLMResponse:
        """Send a new user message and return the model's response."""

    @abstractmethod
    def send_tool_results(self, results: list[ToolResultMessage]) -> LLMResponse:
        """Send tool execution results back and return the model's next response."""


class ProviderError(RuntimeError):
    """Raised when a provider cannot be initialized or fails unrecoverably."""
