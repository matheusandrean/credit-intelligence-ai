"""Portfolio-level endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.llm.tools.registry import call_tool

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("/summary")
def get_portfolio_summary() -> dict:
    """Portfolio-level KPIs: exposure, average PD, Expected Loss, default
    rate, delinquency rate, high-risk population, risk band distribution."""
    result = call_tool("portfolio_summary", {})
    if not result["ok"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result["data"]
