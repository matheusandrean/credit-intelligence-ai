"""Model performance endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.llm.tools.registry import call_tool

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/metrics")
def get_model_metrics() -> dict:
    """Champion model's out-of-time test metrics and calibration method."""
    result = call_tool("get_model_metrics", {})
    if not result["ok"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result["data"]
