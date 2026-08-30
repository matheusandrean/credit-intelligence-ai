"""Headless smoke tests for every Streamlit page using Streamlit's AppTest
framework: each page must render without raising an exception."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.utils.config import PROJECT_ROOT, get_settings


def _artifacts_available() -> bool:
    settings = get_settings()
    return (settings.models_path / "model_metadata.json").exists() and (
        settings.data_path / "processed" / "scored_portfolio.parquet"
    ).exists()


pytestmark = pytest.mark.skipif(
    not _artifacts_available(), reason="trained model artifacts not present"
)

APP_DIR = PROJECT_ROOT / "app"
PAGE_FILES = [APP_DIR / "Home.py", *sorted((APP_DIR / "pages").glob("*.py"))]


@pytest.mark.parametrize("page_path", PAGE_FILES, ids=[p.name for p in PAGE_FILES])
def test_page_runs_without_exception(page_path: Path) -> None:
    at = AppTest.from_file(str(page_path), default_timeout=120)
    at.run()
    assert not at.exception, f"{page_path.name} raised: {at.exception}"
