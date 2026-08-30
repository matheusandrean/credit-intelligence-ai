"""Integration test for the full calibration pipeline (src.models.calibration),
exercising the real champion-selection tie-break rule end to end against
the project's actual trained model artifacts."""

from __future__ import annotations

import pytest

from src.models.calibration import calibrate_champion
from src.utils.config import get_settings


def _artifacts_available() -> bool:
    settings = get_settings()
    return (settings.models_path / "model_metadata.json").exists() and (
        settings.data_path / "processed" / "credit_features.parquet"
    ).exists()


@pytest.mark.skipif(not _artifacts_available(), reason="trained model artifacts not present")
def test_calibrate_champion_end_to_end() -> None:
    report = calibrate_champion()

    assert report["calibration_method"] in {"sigmoid", "isotonic"}
    assert 0 <= report["raw_brier_test"] <= 1
    assert 0 <= report["calibrated_brier_test"] <= 1
    # Calibration should not make the model's Brier score materially worse.
    assert report["calibrated_brier_test"] <= report["raw_brier_test"] * 1.2
    assert "mean_predicted" in report["reliability_calibrated_test"]
    assert "mean_observed" in report["reliability_calibrated_test"]
