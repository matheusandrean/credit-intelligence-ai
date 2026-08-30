"""Build the scored portfolio: PD, illustrative score, risk band and Expected
Loss for every customer.

This is the central artifact consumed by the API, the Streamlit dashboard,
the analytics modules and the LLM agent tools - a single source of truth so
every surface reports the same numbers.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import joblib
import pandas as pd

from src.models.train import split_xy
from src.risk.expected_loss import compute_expected_loss
from src.risk.scoring import pd_to_band
from src.utils.config import PROJECT_ROOT, get_settings
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def build_scored_portfolio() -> pd.DataFrame:
    settings = get_settings()
    models_dir = settings.models_path

    metadata = json.loads((models_dir / "model_metadata.json").read_text(encoding="utf-8"))
    categories = metadata["categories"]
    calibrated_model_path = models_dir / metadata.get(
        "calibrated_model_path", "champion_calibrated.joblib"
    )
    model = joblib.load(calibrated_model_path)

    features_path = settings.data_path / "processed" / "credit_features.parquet"
    df = pd.read_parquet(features_path)

    frames = []
    for split_name in ["train", "validation", "test", "excluded"]:
        subset = df[df["split"] == split_name]
        if subset.empty:
            continue
        x, _ = split_xy(df, split_name, categories)
        pd_hat = model.predict_proba(x)[:, 1]
        scores, bands = pd_to_band(pd_hat)
        el_df = compute_expected_loss(pd_hat, subset["credit_limit"], subset["revolving_balance"])
        scored = pd.DataFrame(
            {
                "customer_id": subset["customer_id"].to_numpy(),
                "observation_date": subset["observation_date"].to_numpy(),
                "origination_date": subset["origination_date"].to_numpy(),
                "split": split_name,
                "pd": pd_hat,
                "score": scores,
                "risk_band": bands,
                "lgd": el_df["lgd"].to_numpy(),
                "ead": el_df["ead"].to_numpy(),
                "expected_loss": el_df["expected_loss"].to_numpy(),
                "target_default_90d": subset["target_default_90d"].to_numpy(),
            }
        )
        frames.append(scored)

    result = pd.concat(frames, ignore_index=True)
    logger.info(
        "scored_portfolio_built",
        n_rows=len(result),
        avg_pd=round(float(result["pd"].mean()), 4),
        total_expected_loss=round(float(result["expected_loss"].sum()), 2),
    )
    return result


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    scored = build_scored_portfolio()
    output_path = settings.data_path / "processed" / "scored_portfolio.parquet"
    scored.to_parquet(output_path, index=False)
    logger.info("scored_portfolio_written", path=str(output_path.relative_to(PROJECT_ROOT)))


if __name__ == "__main__":
    main()
