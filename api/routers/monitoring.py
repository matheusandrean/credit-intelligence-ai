"""Model/data drift monitoring endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.llm.tools.registry import call_tool

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/drift")
def get_drift() -> dict:
    """PSI-based feature drift and predicted-PD drift vs the first available month."""
    result = call_tool("detect_drift", {})
    if not result["ok"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result["data"]
