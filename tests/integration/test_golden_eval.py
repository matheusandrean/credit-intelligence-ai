"""Integration test: the golden dataset evaluation must show perfect tool
selection and evidence grounding for Demo Mode (deterministic, always
runnable without an API key)."""

from __future__ import annotations

import pytest

from src.evaluation.run_golden_eval import run_golden_eval, summarize
from src.utils.config import get_settings


def _artifacts_available() -> bool:
    settings = get_settings()
    return (settings.models_path / "champion_calibrated.joblib").exists() and (
        settings.data_path / "processed" / "scored_portfolio.parquet"
    ).exists()


pytestmark = pytest.mark.skipif(
    not _artifacts_available(), reason="trained model artifacts not present"
)


def test_golden_eval_tool_selection_is_perfect_in_demo_mode() -> None:
    results = run_golden_eval()
    summary = summarize(results)
    failures = [r for r in results if not r.tool_correct]
    assert not failures, f"Tool selection failures: {failures}"
    assert summary["tool_selection_accuracy"] == 1.0


def test_golden_eval_evidence_grounding_is_perfect_in_demo_mode() -> None:
    results = run_golden_eval()
    summary = summarize(results)
    assert summary["evidence_grounding_accuracy"] == 1.0
