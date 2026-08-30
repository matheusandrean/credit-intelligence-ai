"""Delinquency roll-rate analysis.

Computes month-over-month transition matrices between delinquency buckets
(CURRENT, 1-30, 31-60, 61-90, 90+): what share of accounts in bucket X last
month rolled forward to a worse bucket, stayed, or cured back to CURRENT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import DELINQUENCY_BUCKETS


def build_roll_rate_matrix(monthly_performance: pd.DataFrame) -> pd.DataFrame:
    """Return a bucket x bucket transition probability matrix (rows sum to 1).

    Transitions are computed between consecutive MOB snapshots for the same
    customer across the whole panel (pooled, not month-specific) - a
    standard simplification for a portfolio-level roll-rate view.
    """
    panel = monthly_performance.sort_values(["customer_id", "mob"]).copy()
    panel["next_bucket"] = panel.groupby("customer_id")["delinquency_bucket"].shift(-1)
    panel["next_mob"] = panel.groupby("customer_id")["mob"].shift(-1)
    transitions = panel[panel["next_mob"] == panel["mob"] + 1].dropna(subset=["next_bucket"])

    counts = (
        transitions.groupby(["delinquency_bucket", "next_bucket"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=DELINQUENCY_BUCKETS, columns=DELINQUENCY_BUCKETS, fill_value=0)
    )
    row_totals = counts.sum(axis=1).astype(float).replace(0, np.nan)
    matrix = counts.astype(float).div(row_totals, axis=0)
    return matrix.fillna(0.0)


def cure_and_migration_rates(roll_rate_matrix: pd.DataFrame) -> dict:
    """Summarize the matrix into headline cure/migration KPIs."""
    result = {}
    for bucket in DELINQUENCY_BUCKETS:
        if bucket == "CURRENT":
            continue
        if bucket in roll_rate_matrix.index:
            result[f"cure_rate_from_{bucket}"] = float(roll_rate_matrix.loc[bucket, "CURRENT"])
    if "CURRENT" in roll_rate_matrix.index and "1-30" in roll_rate_matrix.columns:
        result["roll_forward_rate_from_current"] = float(roll_rate_matrix.loc["CURRENT", "1-30"])
    return result
