"""Vintage / Months-on-Book (MOB) analysis.

Groups accounts by their origination cohort (the month the account was
opened) and tracks a "bad rate" curve as each cohort ages (MOB), the
classic vintage-curve technique for spotting cohorts that deteriorate
faster than their peers - independent of overall portfolio growth.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_BAD_BUCKETS: tuple[str, ...] = ("61-90", "90+")


def build_vintage_curves(
    monthly_performance: pd.DataFrame,
    customers: pd.DataFrame,
    bad_buckets: tuple[str, ...] = DEFAULT_BAD_BUCKETS,
) -> pd.DataFrame:
    """Return a long-format DataFrame: origination_cohort x mob -> bad_rate.

    `bad_rate` at a given MOB is the share of accounts from that origination
    cohort that are in one of `bad_buckets` at that MOB (a point-in-time
    delinquency rate by age, not a cumulative "ever-bad" rate).
    """
    panel = monthly_performance.merge(
        customers[["customer_id", "origination_date"]], on="customer_id", how="left"
    )
    panel["origination_cohort"] = pd.to_datetime(panel["origination_date"]).dt.to_period("M")
    panel["is_bad"] = panel["delinquency_bucket"].isin(bad_buckets)

    curves = (
        panel.groupby(["origination_cohort", "mob"])
        .agg(n_accounts=("customer_id", "count"), n_bad=("is_bad", "sum"))
        .reset_index()
    )
    curves["bad_rate"] = curves["n_bad"] / curves["n_accounts"]
    curves["origination_cohort"] = curves["origination_cohort"].astype(str)
    return curves.sort_values(["origination_cohort", "mob"]).reset_index(drop=True)


def filter_material_cohorts(vintage_curves: pd.DataFrame, min_accounts: int = 30) -> pd.DataFrame:
    """Drop thin, statistically noisy cohorts (e.g. a handful of long-tenured
    legacy accounts opened years before the observation window) so dashboard
    vintage curves aren't dominated by single-account cohorts."""
    cohort_volume = vintage_curves.groupby("origination_cohort")["n_accounts"].transform("max")
    return vintage_curves[cohort_volume >= min_accounts].reset_index(drop=True)


def cohort_summary(vintage_curves: pd.DataFrame, mob_checkpoint: int = 6) -> pd.DataFrame:
    """Bad rate for every cohort at a fixed MOB checkpoint, for cohort ranking."""
    checkpoint = vintage_curves[vintage_curves["mob"] == mob_checkpoint]
    return checkpoint[["origination_cohort", "n_accounts", "bad_rate"]].sort_values(
        "origination_cohort"
    )
