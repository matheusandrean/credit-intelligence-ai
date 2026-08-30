"""Portfolio stress testing and single-customer what-if simulation.

Both features share the same shock mechanics: perturb a small set of
observable inputs (income, expenses, utilization), propagate the change
through the same feature-engineering formulas used in training, and re-score
with the calibrated champion model. This is an ILLUSTRATIVE simulation of
sensitivity, not a fully re-estimated structural macro model - see
RESPONSIBLE_AI.md for the explicit disclaimer required by the project spec.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import joblib
import numpy as np
import pandas as pd
import yaml

from src.features.build_features import engineer_features
from src.models.train import split_xy
from src.risk.expected_loss import compute_expected_loss, portfolio_expected_loss_summary
from src.risk.scoring import pd_to_band
from src.utils.config import PROJECT_ROOT, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

CONFIG_PATH = PROJECT_ROOT / "configs" / "project_config.yaml"


@dataclass(frozen=True)
class StressScenario:
    name: str
    income_shock_pct: float
    expense_shock_pct: float
    utilization_shock_pp: float


def load_stress_scenarios(config_path: Path = CONFIG_PATH) -> dict[str, StressScenario]:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    scenarios = {}
    for name, params in raw["stress_scenarios"].items():
        scenarios[name] = StressScenario(
            name=name,
            income_shock_pct=params["income_shock_pct"],
            expense_shock_pct=params["expense_shock_pct"],
            utilization_shock_pp=params["utilization_shock_pp"],
        )
    return scenarios


def apply_shock(
    df: pd.DataFrame,
    income_shock_pct: float = 0.0,
    expense_shock_pct: float = 0.0,
    utilization_shock_pp: float = 0.0,
) -> pd.DataFrame:
    """Apply an income/expense/utilization shock and propagate it through the
    same derived-feature formulas used at training time.

    DTI and installment-to-income are both, by construction, proportional to
    `1 / monthly_income` (see src/data/generate_synthetic_credit_data.py), so
    an income shock is propagated to them via that same proportionality
    rather than by re-deriving them from an unobserved raw debt figure.
    """
    out = df.copy()
    old_income = out["monthly_income"].clip(lower=1)
    new_income = old_income * (1 + income_shock_pct)
    income_ratio = old_income / new_income.clip(lower=1)

    out["monthly_income"] = new_income
    out["declared_expenses"] = out["declared_expenses"] * (1 + expense_shock_pct)
    out["average_monthly_spend"] = out["average_monthly_spend"] * (1 + expense_shock_pct)

    out["debt_to_income"] = out["debt_to_income"] * income_ratio
    out["installment_to_income"] = out["installment_to_income"] * income_ratio

    out["credit_utilization"] = np.clip(
        out["credit_utilization"] + utilization_shock_pp / 100.0, 0, 2.0
    )
    out["revolving_balance"] = out["credit_utilization"] * out["credit_limit"]

    out = engineer_features(out)
    return out


def _score(model: object, df: pd.DataFrame, categories: dict[str, list[str]]) -> np.ndarray:
    df_for_split = df.assign(split="all", target_default_90d=0)
    x, _ = split_xy(df_for_split, "all", categories)
    return model.predict_proba(x)[:, 1]


def _load_champion_and_metadata() -> tuple[object, dict]:
    settings = get_settings()
    metadata = json.loads(
        (settings.models_path / "model_metadata.json").read_text(encoding="utf-8")
    )
    model_path = settings.models_path / metadata.get(
        "calibrated_model_path", "champion_calibrated.joblib"
    )
    model = joblib.load(model_path)
    return model, metadata


def run_stress_test(scenario: StressScenario, portfolio: pd.DataFrame) -> dict:
    """Run one stress scenario against the full portfolio and return the
    before/after impact summary."""
    model, metadata = _load_champion_and_metadata()
    categories = metadata["categories"]

    baseline_pd = _score(model, portfolio, categories)
    shocked_df = apply_shock(
        portfolio,
        income_shock_pct=scenario.income_shock_pct,
        expense_shock_pct=scenario.expense_shock_pct,
        utilization_shock_pp=scenario.utilization_shock_pp,
    )
    shocked_pd = _score(model, shocked_df, categories)

    baseline_scores, baseline_bands = pd_to_band(baseline_pd)
    shocked_scores, shocked_bands = pd_to_band(shocked_pd)

    baseline_el = compute_expected_loss(
        baseline_pd, portfolio["credit_limit"], portfolio["revolving_balance"]
    )
    shocked_el = compute_expected_loss(
        shocked_pd, shocked_df["credit_limit"], shocked_df["revolving_balance"]
    )

    return {
        "scenario": scenario.name,
        "shock_parameters": {
            "income_shock_pct": scenario.income_shock_pct,
            "expense_shock_pct": scenario.expense_shock_pct,
            "utilization_shock_pp": scenario.utilization_shock_pp,
        },
        "baseline": {
            "average_pd": float(baseline_pd.mean()),
            **portfolio_expected_loss_summary(baseline_el),
            "band_distribution": pd.Series(baseline_bands).value_counts().to_dict(),
        },
        "stressed": {
            "average_pd": float(shocked_pd.mean()),
            **portfolio_expected_loss_summary(shocked_el),
            "band_distribution": pd.Series(shocked_bands).value_counts().to_dict(),
        },
        "pd_delta": float(shocked_pd.mean() - baseline_pd.mean()),
        "expected_loss_delta": float(
            shocked_el["expected_loss"].sum() - baseline_el["expected_loss"].sum()
        ),
    }


def run_what_if(
    customer_row: pd.DataFrame,
    income_shock_pct: float = 0.0,
    expense_shock_pct: float = 0.0,
    utilization_shock_pp: float = 0.0,
) -> dict:
    """Run a what-if simulation for a single customer row (1-row DataFrame)."""
    model, metadata = _load_champion_and_metadata()
    categories = metadata["categories"]

    baseline_pd = float(_score(model, customer_row, categories)[0])
    shocked_df = apply_shock(
        customer_row,
        income_shock_pct=income_shock_pct,
        expense_shock_pct=expense_shock_pct,
        utilization_shock_pp=utilization_shock_pp,
    )
    shocked_pd = float(_score(model, shocked_df, categories)[0])

    baseline_score, baseline_band = pd_to_band(np.array([baseline_pd]))
    shocked_score, shocked_band = pd_to_band(np.array([shocked_pd]))

    return {
        "baseline": {
            "pd": baseline_pd,
            "score": float(baseline_score[0]),
            "risk_band": str(baseline_band[0]),
            "monthly_income": float(customer_row["monthly_income"].iloc[0]),
            "credit_utilization": float(customer_row["credit_utilization"].iloc[0]),
            "debt_to_income": float(customer_row["debt_to_income"].iloc[0]),
        },
        "simulated": {
            "pd": shocked_pd,
            "score": float(shocked_score[0]),
            "risk_band": str(shocked_band[0]),
            "monthly_income": float(shocked_df["monthly_income"].iloc[0]),
            "credit_utilization": float(shocked_df["credit_utilization"].iloc[0]),
            "debt_to_income": float(shocked_df["debt_to_income"].iloc[0]),
        },
        "pd_delta": shocked_pd - baseline_pd,
        "shock_applied": {
            "income_shock_pct": income_shock_pct,
            "expense_shock_pct": expense_shock_pct,
            "utilization_shock_pp": utilization_shock_pp,
        },
        "is_simulation": True,
    }


def main() -> None:
    from src.utils.logging import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level)

    features_path = settings.data_path / "processed" / "credit_features.parquet"
    portfolio = pd.read_parquet(features_path)

    scenarios = load_stress_scenarios()
    results = {name: run_stress_test(scenario, portfolio) for name, scenario in scenarios.items()}

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "stress_test_report.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    for name, res in results.items():
        logger.info(
            "stress_scenario_complete",
            scenario=name,
            baseline_pd=round(res["baseline"]["average_pd"], 4),
            stressed_pd=round(res["stressed"]["average_pd"], 4),
        )


if __name__ == "__main__":
    main()
