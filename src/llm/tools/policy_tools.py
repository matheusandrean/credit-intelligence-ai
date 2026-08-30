"""Policy-retrieval LLM tool (RAG over knowledge_base/)."""

from __future__ import annotations

from typing import Any

from src.llm.schemas import RetrieveCreditPolicyInput
from src.llm.tools.context import ToolContext
from src.rag.retriever import retrieve


def tool_retrieve_credit_policy(
    input_data: RetrieveCreditPolicyInput, _ctx: ToolContext
) -> dict[str, Any]:
    results = retrieve(input_data.question, top_k=input_data.top_k)
    if not results:
        return {
            "ok": True,
            "data": {
                "evidence": [],
                "note": "No supporting policy evidence was retrieved for this question.",
            },
        }
    return {
        "ok": True,
        "data": {
            "evidence": [
                {
                    "document": r.document,
                    "section": r.section,
                    "text": r.text,
                    "relevance_score": r.relevance_score,
                }
                for r in results
            ]
        },
    }
