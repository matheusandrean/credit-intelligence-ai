"""OpenAI provider adapter (Chat Completions with function calling)."""

from __future__ import annotations

import json
from typing import Any

import openai

from src.llm.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    ProviderError,
    ToolCallRequest,
    ToolResultMessage,
)
from src.llm.tools.registry import ToolSpec

MAX_TOKENS = 2048


def _tool_spec_to_openai(spec: ToolSpec) -> dict[str, Any]:
    schema = spec.input_model.model_json_schema()
    schema.pop("title", None)
    return {
        "type": "function",
        "function": {"name": spec.name, "description": spec.description, "parameters": schema},
    }


class OpenAIProvider(BaseLLMProvider):
    provider_name = "openai"

    def __init__(self, system_prompt: str, tools: list[ToolSpec], api_key: str, model: str):
        super().__init__(system_prompt, tools)
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not set.")
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._openai_tools = [_tool_spec_to_openai(t) for t in tools]
        self._messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    def send_message(self, user_text: str) -> LLMResponse:
        self._messages.append({"role": "user", "content": user_text})
        return self._call()

    def send_tool_results(self, results: list[ToolResultMessage]) -> LLMResponse:
        for r in results:
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": r.call_id,
                    "content": json.dumps(r.content, default=str),
                }
            )
        return self._call()

    def _call(self) -> LLMResponse:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=MAX_TOKENS,
                messages=self._messages,
                tools=self._openai_tools,
            )
        except openai.APIError as exc:
            raise ProviderError(f"OpenAI API error: {exc}") from exc

        message = response.choices[0].message
        self._messages.append(message.model_dump(exclude_none=True))

        tool_calls = [
            ToolCallRequest(
                call_id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments)
            )
            for tc in (message.tool_calls or [])
        ]
        stop_reason = "tool_use" if tool_calls else "end_turn"
        return LLMResponse(
            text=message.content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            raw_usage=response.usage.model_dump() if response.usage else None,
        )
