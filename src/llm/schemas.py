"""Pydantic input/output schemas for every LLM-callable tool.

Every tool the agent can call is typed end-to-end: the LLM must supply
arguments matching a Pydantic input model, and every tool returns a
Pydantic output model that is then JSON-serialized back to the LLM. This is
the mechanism that satisfies the "no arbitrary code / no unrestricted SQL"
requirement - the LLM can only ever invoke one of these fixed, validated
operations (see RESPONSIBLE_AI.md, section on tool safety).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PortfolioSummaryInput(BaseModel):
    pass


class CustomerRiskProfileInput(BaseModel):
    customer_id: str = Field(..., description="Synthetic customer identifier, e.g. CUST_000123")


class CompareSegmentsInput(BaseModel):
    segment_column: str = Field(
        ..., description="Column to segment by, e.g. 'risk_band' or 'age_band'"
    )
    segment_a: str = Field(..., description="First segment value, e.g. 'A'")
    segment_b: str = Field(..., description="Second segment value, e.g. 'D'")


class CalculateDefaultRateInput(BaseModel):
    segment_column: str | None = Field(
        default=None, description="Optional column to break the default rate down by"
    )


class CalculateExpectedLossInput(BaseModel):
    segment_column: str | None = Field(
        default=None, description="Optional column to break Expected Loss down by"
    )


class GetFeatureImportanceInput(BaseModel):
    top_n: int = Field(default=10, ge=1, le=30)


class GetCustomerShapInput(BaseModel):
    customer_id: str = Field(..., description="Synthetic customer identifier")
    top_k: int = Field(default=5, ge=1, le=10)


class RunStressTestInput(BaseModel):
    scenario: str = Field(..., description="One of: baseline, mild, moderate, severe")


class RunWhatIfInput(BaseModel):
    customer_id: str = Field(..., description="Synthetic customer identifier")
    income_shock_pct: float = Field(default=0.0, ge=-0.9, le=2.0)
    expense_shock_pct: float = Field(default=0.0, ge=-0.9, le=2.0)
    utilization_shock_pp: float = Field(default=0.0, ge=-100.0, le=100.0)


class QueryPortfolioInput(BaseModel):
    sql: str = Field(
        ...,
        description=(
            "A read-only SELECT query against the `portfolio` view. "
            "DDL/DML statements are rejected."
        ),
    )


class RetrieveCreditPolicyInput(BaseModel):
    question: str = Field(..., description="Natural-language question about credit policy")
    top_k: int = Field(default=3, ge=1, le=5)


class DetectDriftInput(BaseModel):
    features: list[str] | None = Field(
        default=None, description="Optional subset of features to check for drift"
    )


class GetModelMetricsInput(BaseModel):
    pass


class GenerateExecutiveReportInput(BaseModel):
    pass


# --- Output models ---------------------------------------------------------


class ToolResult(BaseModel):
    """Common envelope: every tool reports whether it succeeded and why not."""

    ok: bool = True
    error: str | None = None
