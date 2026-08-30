"""Anthropic (Claude) provider adapter."""

from __future__ import annotations

import json
from typing import Any

import anthropic

from src.llm.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    ProviderError,
    ToolCallRequest,
    ToolResultMessage,
)
from src.llm.tools.registry import ToolSpec

MAX_TOKENS = 2048


def _tool_spec_to_anthropic(spec: ToolSpec) -> dict[str, Any]:
    schema = spec.input_model.model_json_schema()
    schema.pop("title", None)
    return {"name": spec.name, "description": spec.description, "input_schema": schema}


class AnthropicProvider(BaseLLMProvider):
    provider_name = "anthropic"

    def __init__(self, system_prompt: str, tools: list[ToolSpec], api_key: str, model: str):
        super().__init__(system_prompt, tools)
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set.")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._anthropic_tools = [_tool_spec_to_anthropic(t) for t in tools]
        self._messages: list[dict[str, Any]] = []

    def send_message(self, user_text: str) -> LLMResponse:
        self._messages.append({"role": "user", "content": user_text})
        return self._call()

    def send_tool_results(self, results: list[ToolResultMessage]) -> LLMResponse:
        self._messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.call_id,
                        "content": json.dumps(r.content, default=str),
                    }
                    for r in results
                ],
            }
        )
        return self._call()

    def _call(self) -> LLMResponse:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=MAX_TOKENS,
                system=self.system_prompt,
                messages=self._messages,
                tools=self._anthropic_tools,
            )
        except anthropic.APIError as exc:
            raise ProviderError(f"Anthropic API error: {exc}") from exc

        self._messages.append({"role": "assistant", "content": response.content})

        text_parts = [b.text for b in response.content if b.type == "text"]
        tool_calls = [
            ToolCallRequest(call_id=b.id, name=b.name, arguments=b.input)
            for b in response.content
            if b.type == "tool_use"
        ]
        return LLMResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            raw_usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )
