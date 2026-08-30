"""Pydantic schemas for the /ai/chat endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    provider: str
    tools_called: list[str]
    sources: list[str]
