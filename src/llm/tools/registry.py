"""Central registry of every tool the Credit Intelligence Agent can call.

This is the single place that binds together: the tool name the LLM sees,
its Pydantic input schema (used both for validation and for generating the
provider-specific tool/function-calling spec), a human-readable description
(becomes part of the LLM-facing contract), and the Python implementation.

Adding a new tool means adding one entry here - nothing else in the agent
layer needs to change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from src.llm.schemas import (
    CalculateDefaultRateInput,
    CalculateExpectedLossInput,
    CompareSegmentsInput,
    CustomerRiskProfileInput,
    DetectDriftInput,
    GetCustomerShapInput,
    GetFeatureImportanceInput,
    GetModelMetricsInput,
    PortfolioSummaryInput,
    QueryPortfolioInput,
    RetrieveCreditPolicyInput,
    RunStressTestInput,
    RunWhatIfInput,
)
from src.llm.tools.context import ToolContext, get_tool_context
from src.llm.tools.model_tools import (
    tool_detect_drift,
    tool_get_customer_shap,
    tool_get_feature_importance,
    tool_get_model_metrics,
)
from src.llm.tools.policy_tools import tool_retrieve_credit_policy
from src.llm.tools.portfolio_tools import (
    tool_calculate_default_rate,
    tool_calculate_expected_loss,
    tool_compare_segments,
    tool_customer_risk_profile,
    tool_portfolio_summary,
    tool_query_portfolio,
)
from src.llm.tools.simulation_tools import tool_run_stress_test, tool_run_what_if
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[BaseModel, ToolContext], dict[str, Any]]


TOOL_REGISTRY: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec(
            "portfolio_summary",
            "Get current portfolio-level KPIs: exposure, average PD, Expected Loss, "
            "default rate, delinquency rate, high-risk population, risk band distribution.",
            PortfolioSummaryInput,
            tool_portfolio_summary,
        ),
        ToolSpec(
            "customer_risk_profile",
            "Get a single synthetic customer's PD, score, risk band, Expected Loss and "
            "key behavioral fields by customer_id.",
            CustomerRiskProfileInput,
            tool_customer_risk_profile,
        ),
        ToolSpec(
            "compare_segments",
            "Compare two segments (e.g. two risk bands, two age bands) on PD, score, "
            "Expected Loss, DTI, utilization and observed default rate.",
            CompareSegmentsInput,
            tool_compare_segments,
        ),
        ToolSpec(
            "calculate_default_rate",
            "Calculate the observed default rate, overall or broken down by a segment column.",
            CalculateDefaultRateInput,
            tool_calculate_default_rate,
        ),
        ToolSpec(
            "calculate_expected_loss",
            "Calculate total Expected Loss (PD x LGD x EAD), overall or broken down by a "
            "segment column.",
            CalculateExpectedLossInput,
            tool_calculate_expected_loss,
        ),
        ToolSpec(
            "get_feature_importance",
            "Get the top-N globally most important features driving the credit risk model, "
            "ranked by mean absolute SHAP value.",
            GetFeatureImportanceInput,
            tool_get_feature_importance,
        ),
        ToolSpec(
            "get_customer_shap",
            "Get the top risk-increasing and risk-reducing SHAP factors for one customer's "
            "PD prediction, with real observed feature values.",
            GetCustomerShapInput,
            tool_get_customer_shap,
        ),
        ToolSpec(
            "run_stress_test",
            "Run a portfolio-wide stress scenario (baseline, mild, moderate, severe) and "
            "return the before/after impact on PD, Expected Loss and risk band distribution. "
            "This is a hypothetical simulation, not a forecast.",
            RunStressTestInput,
            tool_run_stress_test,
        ),
        ToolSpec(
            "run_what_if",
            "Run a what-if simulation for one customer (income/expense/utilization shock) "
            "and return the baseline vs simulated PD, score and risk band. Hypothetical only.",
            RunWhatIfInput,
            tool_run_what_if,
        ),
        ToolSpec(
            "query_portfolio",
            "Run a read-only SELECT SQL query against the `portfolio` table (scored "
            "portfolio joined with engineered features) for custom aggregations. "
            "DDL/DML statements are rejected.",
            QueryPortfolioInput,
            tool_query_portfolio,
        ),
        ToolSpec(
            "retrieve_credit_policy",
            "Retrieve the most relevant credit-policy document sections (RAG) for a "
            "natural-language policy question. Always cite document + section.",
            RetrieveCreditPolicyInput,
            tool_retrieve_credit_policy,
        ),
        ToolSpec(
            "detect_drift",
            "Check Population Stability Index (PSI) for key features and the model's "
            "predicted PD distribution over time to detect data/model drift.",
            DetectDriftInput,
            tool_detect_drift,
        ),
        ToolSpec(
            "get_model_metrics",
            "Get the credit risk model's performance metrics (ROC-AUC, KS, Gini, Brier, "
            "calibration method) for the champion model on the out-of-time test set.",
            GetModelMetricsInput,
            tool_get_model_metrics,
        ),
    ]
}


def call_tool(name: str, raw_arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate `raw_arguments` against the tool's schema and execute it.

    Never raises to the caller: validation and execution errors are both
    converted into a structured `{"ok": False, "error": ...}` payload so the
    agent can hand the failure back to the LLM as data.
    """
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}

    try:
        parsed_input = spec.input_model(**raw_arguments)
    except ValidationError as exc:
        return {"ok": False, "error": f"Invalid arguments for {name}: {exc}"}

    try:
        ctx = get_tool_context()
        result = spec.handler(parsed_input, ctx)
    except Exception as exc:  # noqa: BLE001 - tool failures must not crash the agent
        logger.error("tool_execution_failed", tool=name, error=str(exc))
        return {"ok": False, "error": f"Tool '{name}' failed: {exc}"}

    logger.info("tool_called", tool=name, ok=result.get("ok", True))
    return result


def list_tool_specs() -> list[ToolSpec]:
    return list(TOOL_REGISTRY.values())
