"""Deterministic, offline "Demo Mode" provider.

Requires no API key, no network access, and no local LLM runtime. It uses
simple keyword/pattern matching to pick the right tool for a question (the
same tools a real LLM provider would call), then renders the tool's REAL
output into a templated, clearly-labeled response.

This is intentionally NOT a language model: it exists so a recruiter
without any provider credential can still exercise the full tool-calling
pipeline end to end and see genuine computed numbers, never fabricated
ones. See docs/DEMO_MODE.md.
"""

from __future__ import annotations

import re
from typing import Any

from src.llm.providers.base import BaseLLMProvider, LLMResponse, ToolCallRequest, ToolResultMessage
from src.llm.tools.registry import ToolSpec

DEMO_DISCLAIMER = (
    "[Demo Mode: this is a deterministic, rule-based response generated without a "
    "paid LLM provider. It routes to the same tools a real LLM would call and reports "
    "their real output - it does not perform free-form reasoning. Configure "
    "ANTHROPIC_API_KEY, OPENAI_API_KEY, or a local Ollama model for full conversational "
    "analysis.]"
)

_CUSTOMER_ID_RE = re.compile(r"CUST_\d{6}", re.IGNORECASE)
_BAND_RE = re.compile(r"\b([A-E])\b")
_PCT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_PP_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:pp|p\.p\.|percentage points?)")


class DemoProvider(BaseLLMProvider):
    provider_name = "demo"

    def __init__(self, system_prompt: str, tools: list[ToolSpec]):
        super().__init__(system_prompt, tools)
        self._pending_question = ""

    def send_message(self, user_text: str) -> LLMResponse:
        self._pending_question = user_text
        tool_name, arguments = self._classify_intent(user_text)
        return LLMResponse(
            text=None,
            tool_calls=[ToolCallRequest(call_id="demo-1", name=tool_name, arguments=arguments)],
            stop_reason="tool_use",
        )

    def send_tool_results(self, results: list[ToolResultMessage]) -> LLMResponse:
        rendered = "\n\n".join(self._render_result(r.name, r.content) for r in results)
        text = f"{DEMO_DISCLAIMER}\n\n{rendered}"
        return LLMResponse(text=text, tool_calls=[], stop_reason="end_turn")

    # --- Intent classification --------------------------------------------

    def _classify_intent(self, question: str) -> tuple[str, dict[str, Any]]:
        q = question.lower()
        customer_match = _CUSTOMER_ID_RE.search(question)
        customer_id = customer_match.group(0).upper() if customer_match else None

        if any(k in q for k in ["policy", "política", "politica", "policies"]):
            return "retrieve_credit_policy", {"question": question}

        if any(k in q for k in ["what if", "what-if", "simulat"]) and customer_id:
            return "run_what_if", {
                "customer_id": customer_id,
                **self._extract_shock_params(q),
            }

        if any(k in q for k in ["stress", "scenario", "cenário", "cenario"]):
            scenario = "severe"
            for name in ["baseline", "mild", "moderate", "severe"]:
                if name in q:
                    scenario = name
                    break
            return "run_stress_test", {"scenario": scenario}

        if customer_id and any(
            k in q for k in ["why", "explain", "factor", "driver", "shap", "porque", "por que"]
        ):
            return "get_customer_shap", {"customer_id": customer_id}

        if customer_id:
            return "customer_risk_profile", {"customer_id": customer_id}

        if any(k in q for k in ["compare", "comparar", "compare risk bands", "compare bands"]):
            bands = _BAND_RE.findall(question.upper())
            band_a, band_b = (bands + ["A", "D"])[:2]
            return "compare_segments", {
                "segment_column": "risk_band",
                "segment_a": band_a,
                "segment_b": band_b,
            }

        if any(k in q for k in ["drift", "psi", "deteriorat", "deterior"]):
            return "detect_drift", {}

        if any(
            k in q for k in ["driving", "drivers", "factors", "fatores", "contribute", "importan"]
        ):
            return "get_feature_importance", {"top_n": 10}

        if any(k in q for k in ["default rate", "taxa de default", "taxa de inadimpl"]):
            return "calculate_default_rate", {"segment_column": "risk_band"}

        if any(k in q for k in ["expected loss", "perda esperada", "el "]):
            return "calculate_expected_loss", {"segment_column": "risk_band"}

        if any(k in q for k in ["auc", "ks", "gini", "model metric", "métrica", "performance"]):
            return "get_model_metrics", {}

        return "portfolio_summary", {}

    def _extract_shock_params(self, q: str) -> dict[str, float]:
        params: dict[str, float] = {
            "income_shock_pct": 0.0,
            "expense_shock_pct": 0.0,
            "utilization_shock_pp": 0.0,
        }
        pct_matches = [float(m) / 100 for m in _PCT_RE.findall(q)]
        pp_matches = [float(m) for m in _PP_RE.findall(q)]

        if "income" in q or "renda" in q:
            for m in pct_matches:
                if m < 0:
                    params["income_shock_pct"] = m
        if "expense" in q or "despesa" in q or "gasto" in q:
            for m in pct_matches:
                if m > 0:
                    params["expense_shock_pct"] = m
        if pp_matches:
            params["utilization_shock_pp"] = pp_matches[0]
        return params

    # --- Rendering -----------------------------------------------------------

    def _render_result(self, tool_name: str, content: dict[str, Any]) -> str:
        if not content.get("ok", True):
            return f"Tool `{tool_name}` could not produce evidence: {content.get('error')}"
        data = content.get("data", {})
        return f"Tool called: `{tool_name}`\nResult:\n{_format_data(data)}"


def _format_data(data: Any, indent: int = 0) -> str:
    import json

    return json.dumps(data, indent=2, default=str)
