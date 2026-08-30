"""Integration test: every SQL report in sql/ must execute successfully
against the real generated parquet data and return rows."""

from __future__ import annotations

import pytest

from src.analytics.run_sql_reports import run_all_reports
from src.utils.config import get_settings


def _artifacts_available() -> bool:
    settings = get_settings()
    return (settings.data_path / "processed" / "scored_portfolio.parquet").exists() and (
        settings.data_path / "raw" / "monthly_performance.parquet"
    ).exists()


@pytest.mark.skipif(not _artifacts_available(), reason="generated data not present")
def test_all_sql_reports_execute_and_return_rows() -> None:
    row_counts = run_all_reports()
    assert len(row_counts) >= 6
    for name, count in row_counts.items():
        assert count > 0, f"{name} returned zero rows"
