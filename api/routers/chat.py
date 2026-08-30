"""AI Credit Analyst chat endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_agent
from api.schemas.chat import ChatRequest, ChatResponse
from src.agents.credit_intelligence_agent import CreditIntelligenceAgent

router = APIRouter(tags=["ai"])


@router.post("/ai/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, agent: CreditIntelligenceAgent = Depends(get_agent)) -> ChatResponse:
    response = agent.ask(payload.question)
    return ChatResponse(
        answer=response.answer,
        provider=response.provider,
        tools_called=response.tools_called,
        sources=response.sources,
    )
