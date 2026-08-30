"""SHAP-based explainability for the credit risk models.

Supports both champion candidates:
- LightGBM: `shap.TreeExplainer` directly on the fitted booster.
- Logistic Regression pipeline: `shap.LinearExplainer` on the
  post-preprocessing feature space (after the ColumnTransformer), since SHAP
  needs a numeric feature matrix and named coefficients to attribute.

Produces both per-customer local explanations (top risk-increasing /
risk-reducing factors) and a global feature-importance summary. The LLM
layer is only ever allowed to *narrate* these numbers - it must never
invent a driver that is not present in the SHAP output (see
src/llm/tools/explain_customer.py and RESPONSIBLE_AI.md).
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.models.train import load_dataset, split_xy
from src.utils.config import PROJECT_ROOT, get_settings
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


@dataclass
class LocalExplanation:
    customer_id: str
    base_value: float
    predicted_pd: float
    top_risk_increasing: list[dict[str, Any]]
    top_risk_reducing: list[dict[str, Any]]


class ShapExplainer:
    """Wraps a fitted champion model with a uniform SHAP interface."""

    def __init__(self, champion_name: str, model: Any, background: pd.DataFrame):
        self.champion_name = champion_name
        self.model = model
        if champion_name == "lightgbm":
            self._explainer = shap.TreeExplainer(model)
            self.feature_names = list(background.columns)
            self._is_linear = False
        else:
            preprocessor: Any = (
                model.named_steps["preprocess"] if isinstance(model, Pipeline) else None
            )
            if preprocessor is None:
                raise ValueError(f"Unsupported model type for SHAP: {champion_name}")
            linear_model = model.named_steps["model"]
            transformed_background = self._transform(preprocessor, background)
            self._explainer = shap.LinearExplainer(linear_model, transformed_background)
            self.feature_names = list(preprocessor.get_feature_names_out())
            self._preprocessor = preprocessor
            self._is_linear = True

    @staticmethod
    def _transform(preprocessor: Any, x: pd.DataFrame) -> np.ndarray:
        transformed = preprocessor.transform(x)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return np.asarray(transformed)

    def shap_values(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (shap_values, feature_display_values, base_value) for rows in `x`."""
        if self._is_linear:
            transformed = self._transform(self._preprocessor, x)
            values = self._explainer.shap_values(transformed)
            base_value = float(np.atleast_1d(self._explainer.expected_value)[0])
            return values, transformed, base_value

        raw_values = self._explainer.shap_values(x)
        # LightGBM binary classifier: newer SHAP may return a list [class0, class1]
        # or an array with an extra trailing class dimension; normalize to the
        # positive-class contribution matrix.
        if isinstance(raw_values, list):
            values = raw_values[1]
            base_value = float(np.atleast_1d(self._explainer.expected_value)[1])
        elif isinstance(raw_values, np.ndarray) and raw_values.ndim == 3:
            values = raw_values[:, :, 1]
            base_value = float(np.atleast_1d(self._explainer.expected_value)[1])
        else:
            values = raw_values
            base_value = float(np.atleast_1d(self._explainer.expected_value)[0])
        return values, x.to_numpy(), base_value

    def _resolve_base_feature(self, feature_name: str, x_row: pd.DataFrame) -> tuple[str, Any]:
        """Map a SHAP feature name back to (raw_column_name, raw_value).

        SHAP operates on the post-preprocessing feature space (standardized
        numerics, one-hot dummies per category), which is not meaningful to
        show a credit analyst. Numeric features map 1:1 to a raw column;
        one-hot categorical dummies are collapsed back to their base column
        (e.g. `payment_history_Poor` -> `payment_history`) so each category
        is reported once with its actual observed value.
        """
        name = feature_name.removeprefix("numeric__").removeprefix("categorical__")
        if name in x_row.columns:
            return name, _safe_scalar(x_row.iloc[0][name])
        for col in x_row.columns:
            if name.startswith(f"{col}_"):
                return col, _safe_scalar(x_row.iloc[0][col])
        return name, None

    def explain_customer(
        self, customer_id: str, x_row: pd.DataFrame, predicted_pd: float, top_k: int = 5
    ) -> LocalExplanation:
        values, _display_values, base_value = self.shap_values(x_row)
        row_values = values[0]

        aggregated: dict[str, dict[str, Any]] = {}
        for i in range(len(self.feature_names)):
            base_col, raw_value = self._resolve_base_feature(self.feature_names[i], x_row)
            entry = aggregated.setdefault(base_col, {"value": raw_value, "shap_value": 0.0})
            entry["shap_value"] += float(row_values[i])

        contributions = [{"feature": feature, **data} for feature, data in aggregated.items()]
        contributions.sort(key=lambda c: c["shap_value"], reverse=True)
        top_increasing = [c for c in contributions if c["shap_value"] > 0][:top_k]
        top_reducing = list(reversed([c for c in contributions if c["shap_value"] < 0][-top_k:]))

        return LocalExplanation(
            customer_id=customer_id,
            base_value=base_value,
            predicted_pd=float(predicted_pd),
            top_risk_increasing=top_increasing,
            top_risk_reducing=top_reducing,
        )

    def global_importance(self, x_sample: pd.DataFrame) -> pd.DataFrame:
        values, _, _ = self.shap_values(x_sample)
        mean_abs = np.abs(values).mean(axis=0)
        display_names = [
            n.removeprefix("numeric__").removeprefix("categorical__") for n in self.feature_names
        ]
        importance = pd.DataFrame({"feature": display_names, "mean_abs_shap": mean_abs})
        return (
            importance.groupby("feature", as_index=False)["mean_abs_shap"]
            .sum()
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, np.floating | np.integer):
        return value.item()
    return value


def load_model_metadata() -> dict:
    settings = get_settings()
    return json.loads((settings.models_path / "model_metadata.json").read_text(encoding="utf-8"))


def build_default_explainer(
    background_size: int = 300,
) -> tuple[ShapExplainer, pd.DataFrame, dict]:
    """Build a ShapExplainer for the raw champion model using a train-split background.

    Returns (explainer, full_feature_dataset, model_metadata). SHAP explains
    the raw (pre-calibration) champion model's decision function; the PD
    shown to users elsewhere in the app comes from the calibrated model, but
    calibration (isotonic/Platt) is monotonic so the *ranking and direction*
    of drivers SHAP reports remain valid interpretations of what pushed risk
    up or down.
    """
    metadata = load_model_metadata()
    champion_name = metadata["champion_model"]
    categories = metadata["categories"]

    settings = get_settings()
    model = joblib.load(settings.models_path / f"{champion_name}.joblib")

    df = load_dataset()
    x_train, _ = split_xy(df, "train", categories)
    background = x_train.sample(min(background_size, len(x_train)), random_state=42)

    explainer = ShapExplainer(champion_name, model, background)
    return explainer, df, metadata


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    explainer, df, metadata = build_default_explainer()
    categories = metadata["categories"]

    x_test, _ = split_xy(df, "test", categories)
    sample = x_test.sample(min(1000, len(x_test)), random_state=7)

    global_importance = explainer.global_importance(sample)
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    global_importance.to_csv(reports_dir / "shap_global_importance.csv", index=False)

    test_customer_ids = df.loc[x_test.index, "customer_id"].reset_index(drop=True)
    examples = []
    for i in range(min(5, len(x_test))):
        row = x_test.iloc[[i]]
        pd_hat = float(explainer.model.predict_proba(row)[0, 1])
        explanation = explainer.explain_customer(
            str(test_customer_ids.iloc[i]), row, predicted_pd=pd_hat
        )
        examples.append(
            {
                "customer_id": explanation.customer_id,
                "top_risk_increasing": explanation.top_risk_increasing,
                "top_risk_reducing": explanation.top_risk_reducing,
            }
        )
    (reports_dir / "shap_example_explanations.json").write_text(
        json.dumps(examples, indent=2), encoding="utf-8"
    )

    logger.info(
        "shap_explainability_complete",
        champion_model=metadata["champion_model"],
        top_feature=global_importance.iloc[0]["feature"],
    )


if __name__ == "__main__":
    main()
