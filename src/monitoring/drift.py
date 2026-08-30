"""Population Stability Index (PSI) and model performance monitoring.

Thresholds (PSI < 0.10 stable, 0.10-0.25 monitor, > 0.25 potential
significant drift) follow common industry convention but are explicitly
DEMONSTRATIVE - see configs/project_config.yaml and RESPONSIBLE_AI.md; a
real deployment must calibrate these to its own governance framework.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.metrics import compute_classification_metrics
from src.utils.config import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "configs" / "project_config.yaml"


def load_drift_thresholds(config_path: Path = CONFIG_PATH) -> tuple[float, float]:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    drift_cfg = raw["drift"]
    return drift_cfg["psi_stable_threshold"], drift_cfg["psi_monitor_threshold"]


def population_stability_index(
    reference: pd.Series | np.ndarray, current: pd.Series | np.ndarray, n_buckets: int = 10
) -> float:
    """Compute PSI between a reference and current distribution of a numeric
    feature, using reference-derived quantile bucket edges."""
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()
    if reference.empty or current.empty:
        return 0.0

    quantiles = np.linspace(0, 1, n_buckets + 1)
    edges = np.unique(reference.quantile(quantiles).to_numpy())
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts = pd.cut(reference, bins=edges).value_counts(sort=False)
    cur_counts = pd.cut(current, bins=edges).value_counts(sort=False)

    ref_pct = (ref_counts / ref_counts.sum()).replace(0, 1e-6)
    cur_pct = (cur_counts / cur_counts.sum()).replace(0, 1e-6)

    psi = float(((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum())
    return psi


def psi_status(
    psi: float, stable_threshold: float | None = None, monitor_threshold: float | None = None
) -> str:
    if stable_threshold is None or monitor_threshold is None:
        stable_threshold, monitor_threshold = load_drift_thresholds()
    if psi < stable_threshold:
        return "Stable"
    if psi < monitor_threshold:
        return "Monitor"
    return "Potential significant drift"


def feature_drift_report(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    stable_threshold, monitor_threshold = load_drift_thresholds()
    rows = []
    for feature in features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        psi = population_stability_index(reference_df[feature], current_df[feature])
        rows.append(
            {
                "feature": feature,
                "psi": round(psi, 4),
                "status": psi_status(psi, stable_threshold, monitor_threshold),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def model_performance_over_time(scored: pd.DataFrame) -> pd.DataFrame:
    """Monthly ROC-AUC/KS/Gini/PSI-of-predicted-PD, using the earliest
    available month's predicted-PD distribution as the drift reference."""
    scored = scored.copy()
    scored["month"] = pd.to_datetime(scored["observation_date"]).dt.to_period("M")
    months = sorted(scored["month"].unique())
    if not months:
        return pd.DataFrame()

    reference_pd = scored.loc[scored["month"] == months[0], "pd"]

    rows = []
    for month in months:
        subset = scored[scored["month"] == month]
        if subset["target_default_90d"].nunique() < 2:
            metrics = None
        else:
            metrics = compute_classification_metrics(
                subset["target_default_90d"].to_numpy(), subset["pd"].to_numpy()
            )
        psi = population_stability_index(reference_pd, subset["pd"])
        rows.append(
            {
                "month": str(month),
                "n_accounts": int(len(subset)),
                "roc_auc": metrics.roc_auc if metrics else None,
                "ks_statistic": metrics.ks_statistic if metrics else None,
                "gini": metrics.gini if metrics else None,
                "average_pd": float(subset["pd"].mean()),
                "observed_default_rate": float(subset["target_default_90d"].mean()),
                "psi_vs_first_month": round(psi, 4),
                "psi_status": psi_status(psi),
            }
        )
    return pd.DataFrame(rows)
