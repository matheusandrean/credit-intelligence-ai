"""Integration tests for the Portfolio Executive Report generator."""

from __future__ import annotations

import pytest

from src.llm.tools.registry import call_tool
from src.reporting.executive_report import generate_executive_report
from src.utils.config import get_settings


def _artifacts_available() -> bool:
    settings = get_settings()
    return (settings.models_path / "champion_calibrated.joblib").exists() and (
        settings.data_path / "processed" / "scored_portfolio.parquet"
    ).exists()


pytestmark = pytest.mark.skipif(
    not _artifacts_available(), reason="trained model artifacts not present"
)

EXPECTED_SECTIONS = [
    "## Portfolio Overview",
    "## Risk Movement",
    "## Main Drivers",
    "## Deterioration Signals",
    "## Stress Testing",
    "## Model Health",
    "## Points Requiring Human Attention",
]


def test_report_contains_all_required_sections() -> None:
    report = generate_executive_report()
    for section in EXPECTED_SECTIONS:
        assert section in report


def test_report_is_grounded_in_real_numbers() -> None:
    report = generate_executive_report()
    assert "Total customers:" in report
    assert "%" in report  # PD/rate figures present


def test_report_includes_decision_support_disclaimer() -> None:
    report = generate_executive_report()
    assert "decision support only" in report.lower()


def test_generate_executive_report_tool() -> None:
    result = call_tool("generate_executive_report", {})
    assert result["ok"] is True
    assert "## Portfolio Overview" in result["data"]["report_markdown"]
