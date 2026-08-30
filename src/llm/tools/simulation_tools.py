"""Simulation LLM tools: portfolio stress testing and single-customer what-if.

Both are explicitly hypothetical: every response is tagged with
`is_simulation: true` and the tools never write back into the scored
portfolio or trigger any real action.
"""

from __future__ import annotations

from typing import Any

from src.llm.schemas import RunStressTestInput, RunWhatIfInput
from src.llm.tools.context import ToolContext
from src.risk.stress_testing import load_stress_scenarios, run_stress_test, run_what_if


def tool_run_stress_test(input_data: RunStressTestInput, ctx: ToolContext) -> dict[str, Any]:
    scenarios = load_stress_scenarios()
    scenario = scenarios.get(input_data.scenario.lower())
    if scenario is None:
        return {
            "ok": False,
            "error": f"Unknown scenario '{input_data.scenario}'. Valid: {sorted(scenarios)}",
        }
    result = run_stress_test(scenario, ctx.features)
    result["is_simulation"] = True
    return {"ok": True, "data": result}


def tool_run_what_if(input_data: RunWhatIfInput, ctx: ToolContext) -> dict[str, Any]:
    row = ctx.get_customer_row(input_data.customer_id)
    if row is None:
        return {"ok": False, "error": f"Customer {input_data.customer_id} not found."}

    result = run_what_if(
        row,
        income_shock_pct=input_data.income_shock_pct,
        expense_shock_pct=input_data.expense_shock_pct,
        utilization_shock_pp=input_data.utilization_shock_pp,
    )
    return {"ok": True, "data": result}
