"""Executive report generation tool."""

from __future__ import annotations

from typing import Any

from src.llm.schemas import GenerateExecutiveReportInput
from src.llm.tools.context import ToolContext
from src.reporting.executive_report import generate_executive_report


def tool_generate_executive_report(
    _input: GenerateExecutiveReportInput, _ctx: ToolContext
) -> dict[str, Any]:
    report_markdown = generate_executive_report()
    return {"ok": True, "data": {"report_markdown": report_markdown}}
