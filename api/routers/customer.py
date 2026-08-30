"""Customer-level endpoints: profile, risk, and SHAP explanation."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.llm.tools.context import get_tool_context
from src.llm.tools.json_utils import to_json_safe
from src.llm.tools.registry import call_tool

router = APIRouter(prefix="/customer", tags=["customer"])


@router.get("/{customer_id}")
def get_customer(customer_id: str) -> dict:
    """Basic observed customer profile (raw/engineered fields, no model output)."""
    ctx = get_tool_context()
    row = ctx.get_customer_row(customer_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found.")
    return to_json_safe(row.iloc[0].to_dict())


@router.get("/{customer_id}/risk")
def get_customer_risk(customer_id: str) -> dict:
    """Model output for this customer: PD, illustrative score, risk band,
    Expected Loss and exposure."""
    result = call_tool("customer_risk_profile", {"customer_id": customer_id})
    if not result["ok"]:
        raise HTTPException(status_code=404, detail=result["error"])
    return result["data"]


@router.get("/{customer_id}/explanation")
def get_customer_explanation(customer_id: str, top_k: int = Query(default=5, ge=1, le=10)) -> dict:
    """SHAP-based explanation: top risk-increasing and risk-reducing factors."""
    result = call_tool("get_customer_shap", {"customer_id": customer_id, "top_k": top_k})
    if not result["ok"]:
        raise HTTPException(status_code=404, detail=result["error"])
    return result["data"]
