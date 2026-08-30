"""Probability calibration for the champion credit risk model.

Raw classifier scores (especially from tree ensembles) are often poorly
calibrated: a predicted "20% PD" bucket may not actually default 20% of the
time. This module fits both Platt scaling (sigmoid) and isotonic
regression on the validation split (never train, to avoid overfitting the
calibration itself), picks whichever improves the Brier score most on
validation, and reports the raw-vs-calibrated comparison plus reliability
curves on the untouched OOT test split.

The calibrated model - not the raw classifier - is what the rest of the
application (API, dashboard, agent tools) uses as "the PD model", because
Probability of Default is meant to be read as a real probability (see
MODEL_CARD.md).
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

from src.models.metrics import reliability_curve
from src.models.train import load_dataset, split_xy
from src.utils.config import PROJECT_ROOT, get_settings
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def calibrate_champion() -> dict:
    settings = get_settings()
    models_dir = settings.models_path
    metadata = json.loads((models_dir / "model_metadata.json").read_text(encoding="utf-8"))
    champion_name = metadata["champion_model"]
    categories = metadata["categories"]

    model_path = models_dir / f"{champion_name}.joblib"
    champion_model = joblib.load(model_path)

    df = load_dataset()
    x_train, y_train = split_xy(df, "train", categories)
    x_val, y_val = split_xy(df, "validation", categories)
    x_test, y_test = split_xy(df, "test", categories)

    def predict(model, x):
        return model.predict_proba(x)[:, 1]

    raw_test_prob = predict(champion_model, x_test)
    raw_brier_test = float(brier_score_loss(y_test, raw_test_prob))

    candidates: dict[str, tuple[float, object]] = {}
    for method in ["sigmoid", "isotonic"]:
        calibrated = CalibratedClassifierCV(FrozenEstimator(champion_model), method=method)
        calibrated.fit(x_val, y_val)
        val_brier = float(brier_score_loss(y_val, predict(calibrated, x_val)))
        logger.info("calibration_candidate", method=method, val_brier=val_brier)
        candidates[method] = (val_brier, calibrated)

    # Prefer Platt/sigmoid scaling by default: it is a smooth, strictly
    # monotonic transform, which matters for this platform's what-if and
    # stress-testing tools (a small input change should always be able to
    # move the PD). Isotonic regression is a step function and can produce
    # flat plateaus - especially in the sparse low-probability tail - that
    # silently absorb small input changes. We only switch to isotonic when
    # it improves validation Brier score by a material margin (>5% relative),
    # since a plateau is an acceptable trade-off for a genuinely better-
    # calibrated model, documented in MODEL_CARD.md.
    sigmoid_brier, _ = candidates["sigmoid"]
    isotonic_brier, _ = candidates["isotonic"]
    best_method = "isotonic" if isotonic_brier < sigmoid_brier * 0.95 else "sigmoid"
    best_model = candidates[best_method][1]

    calibrated_test_prob = predict(best_model, x_test)
    calibrated_brier_test = float(brier_score_loss(y_test, calibrated_test_prob))

    output_path = models_dir / "champion_calibrated.joblib"
    joblib.dump(best_model, output_path)

    report = {
        "champion_model": champion_name,
        "calibration_method": best_method,
        "raw_brier_test": raw_brier_test,
        "calibrated_brier_test": calibrated_brier_test,
        "improvement": raw_brier_test - calibrated_brier_test,
        "reliability_raw_test": reliability_curve(y_test, raw_test_prob),
        "reliability_calibrated_test": reliability_curve(y_test, calibrated_test_prob),
    }

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    metadata["calibration_method"] = best_method
    metadata["calibrated_model_path"] = "champion_calibrated.joblib"
    (models_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    logger.info(
        "calibration_complete",
        method=best_method,
        raw_brier_test=round(raw_brier_test, 4),
        calibrated_brier_test=round(calibrated_brier_test, 4),
    )
    return report


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    calibrate_champion()


if __name__ == "__main__":
    main()
