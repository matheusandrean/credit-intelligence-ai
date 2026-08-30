"""What-if and stress-test simulation endpoints. All outputs are explicitly
hypothetical - see the `is_simulation` flag in every response."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.llm.schemas import RunStressTestInput, RunWhatIfInput
from src.llm.tools.registry import call_tool

router = APIRouter(tags=["simulation"])


@router.post("/simulation/what-if")
def what_if(payload: RunWhatIfInput) -> dict:
    result = call_tool("run_what_if", payload.model_dump())
    if not result["ok"]:
        raise HTTPException(status_code=404, detail=result["error"])
    return result["data"]


@router.post("/stress-test")
def stress_test(payload: RunStressTestInput) -> dict:
    result = call_tool("run_stress_test", payload.model_dump())
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result["data"]
