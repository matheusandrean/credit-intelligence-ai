"""Provider factory: builds the configured LLM provider from settings.

`LLM_PROVIDER` selects one of: anthropic | openai | ollama | demo.
Falls back to Demo Mode with a clear log message if a paid provider is
selected but its credential is missing, so the app never hard-crashes for a
recruiter without an API key.
"""

from __future__ import annotations

from src.llm.providers.anthropic_provider import AnthropicProvider
from src.llm.providers.base import BaseLLMProvider, ProviderError
from src.llm.providers.demo_provider import DemoProvider
from src.llm.providers.ollama_provider import OllamaProvider
from src.llm.providers.openai_provider import OpenAIProvider
from src.llm.system_prompt import SYSTEM_PROMPT
from src.llm.tools.registry import list_tool_specs
from src.utils.config import Settings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_provider(settings: Settings | None = None) -> BaseLLMProvider:
    settings = settings or get_settings()
    tools = list_tool_specs()
    provider_name = settings.llm_provider.lower()

    try:
        if provider_name == "anthropic":
            return AnthropicProvider(
                SYSTEM_PROMPT, tools, settings.anthropic_api_key or "", settings.anthropic_model
            )
        if provider_name == "openai":
            return OpenAIProvider(
                SYSTEM_PROMPT, tools, settings.openai_api_key or "", settings.openai_model
            )
        if provider_name == "ollama":
            return OllamaProvider(
                SYSTEM_PROMPT, tools, settings.ollama_base_url, settings.ollama_model
            )
    except ProviderError as exc:
        logger.warning("llm_provider_fallback_to_demo", requested=provider_name, reason=str(exc))
        return DemoProvider(SYSTEM_PROMPT, tools)

    return DemoProvider(SYSTEM_PROMPT, tools)
