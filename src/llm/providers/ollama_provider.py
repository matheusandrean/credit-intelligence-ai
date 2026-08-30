"""Ollama provider adapter (local models, no API key required).

Uses Ollama's `/api/chat` endpoint, which supports OpenAI-style tool
calling for compatible models (e.g. llama3.1, qwen2.5). Requires a locally
running Ollama server - see docs/DEMO_MODE.md for setup instructions.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from src.llm.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    ProviderError,
    ToolCallRequest,
    ToolResultMessage,
)
from src.llm.tools.registry import ToolSpec

REQUEST_TIMEOUT_SECONDS = 120


def _tool_spec_to_ollama(spec: ToolSpec) -> dict[str, Any]:
    schema = spec.input_model.model_json_schema()
    schema.pop("title", None)
    return {
        "type": "function",
        "function": {"name": spec.name, "description": spec.description, "parameters": schema},
    }


class OllamaProvider(BaseLLMProvider):
    provider_name = "ollama"

    def __init__(self, system_prompt: str, tools: list[ToolSpec], base_url: str, model: str):
        super().__init__(system_prompt, tools)
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._ollama_tools = [_tool_spec_to_ollama(t) for t in tools]
        self._messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    def send_message(self, user_text: str) -> LLMResponse:
        self._messages.append({"role": "user", "content": user_text})
        return self._call()

    def send_tool_results(self, results: list[ToolResultMessage]) -> LLMResponse:
        for r in results:
            self._messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(r.content, default=str),
                }
            )
        return self._call()

    def _call(self) -> LLMResponse:
        try:
            response = httpx.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": self._messages,
                    "tools": self._ollama_tools,
                    "stream": False,
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Could not reach local Ollama server at {self._base_url}: {exc}"
            ) from exc

        payload = response.json()
        message = payload.get("message", {})
        self._messages.append(message)

        raw_tool_calls = message.get("tool_calls") or []
        tool_calls = [
            ToolCallRequest(
                call_id=str(i),
                name=tc["function"]["name"],
                arguments=tc["function"].get("arguments", {}),
            )
            for i, tc in enumerate(raw_tool_calls)
        ]
        stop_reason = "tool_use" if tool_calls else "end_turn"
        return LLMResponse(
            text=message.get("content"), tool_calls=tool_calls, stop_reason=stop_reason
        )
