"""Shared FastAPI dependencies: a process-wide cached agent instance."""

from __future__ import annotations

from functools import lru_cache

from src.agents.credit_intelligence_agent import CreditIntelligenceAgent, get_default_agent


@lru_cache
def get_agent() -> CreditIntelligenceAgent:
    return get_default_agent()
