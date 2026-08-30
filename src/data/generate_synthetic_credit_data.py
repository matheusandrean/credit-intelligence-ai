"""Synthetic credit portfolio generator.

Generates a fully synthetic ~100k-customer credit portfolio with plausible
statistical relationships between risk drivers (DTI, utilization, payment
history, previous default, balance/utilization/delinquency trends) and a
90-day default target, plus a monthly delinquency-bucket performance panel
used for vintage / MOB / roll-rate analytics.

ZERO real personal data is used or required. All identifiers are synthetic
sequential IDs (CUST_000001, ...). No protected/sensitive characteristics
(race, religion, gender, etc.) are generated or used - see
`src/data/schema.py::PROHIBITED_PROTECTED_ATTRIBUTES` and RESPONSIBLE_AI.md.

Run as a script:
    python -m src.data.generate_synthetic_credit_data
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.schema import (
    AGE_BANDS,
    DELINQUENCY_BUCKETS,
    PAYMENT_HISTORY_LEVELS,
)
from src.utils.config import PROJECT_ROOT, get_settings
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

CONFIG_PATH = PROJECT_ROOT / "configs" / "project_config.yaml"


@dataclass(frozen=True)
class GenerationConfig:
    n_customers: int
    random_seed: int
    start_date: str
    end_date: str
    target_default_rate: float = 0.075


def load_generation_config(config_path: Path = CONFIG_PATH) -> GenerationConfig:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    data_cfg = raw["data"]
    target_rate = raw.get("target", {}).get("target_portfolio_default_rate", 0.075)
    return GenerationConfig(
        n_customers=data_cfg["n_customers"],
        random_seed=data_cfg["random_seed"],
        start_date=data_cfg["start_date"],
        end_date=data_cfg["end_date"],
        target_default_rate=target_rate,
    )


def _calibrate_intercept(
    latent_score: np.ndarray, target_rate: float, slope: float = 0.85
) -> float:
    """Binary-search an intercept so mean(sigmoid(intercept + slope*latent)) == target_rate."""
    lo, hi = -12.0, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = _sigmoid(mid + slope * latent_score).mean()
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    if std == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def _inject_missing(rng: np.random.Generator, series: pd.Series, missing_rate: float) -> pd.Series:
    mask = rng.random(len(series)) < missing_rate
    out = series.copy()
    out[mask] = np.nan
    return out


def _inject_outliers(
    rng: np.random.Generator,
    series: pd.Series,
    outlier_rate: float,
    multiplier_range: tuple[float, float],
) -> pd.Series:
    mask = rng.random(len(series)) < outlier_rate
    multipliers = rng.uniform(*multiplier_range, size=len(series))
    out = series.copy()
    out[mask] = out[mask] * multipliers[mask]
    return out


def generate_customer_snapshot(cfg: GenerationConfig) -> pd.DataFrame:
    """Generate the customer-level snapshot table used for credit risk modeling.

    Returns a DataFrame with one row per synthetic customer including the
    engineered risk drivers and the `target_default_90d` label. Statistical
    noise, missing values, outliers and mild multicollinearity are injected
    deliberately so the modeling problem is realistic rather than trivially
    separable.
    """
    rng = np.random.default_rng(cfg.random_seed)
    n = cfg.n_customers

    logger.info("generating_customer_snapshot", n_customers=n, seed=cfg.random_seed)

    customer_id = np.array([f"CUST_{i:06d}" for i in range(1, n + 1)])

    # --- Temporal structure -------------------------------------------------
    start = pd.Timestamp(cfg.start_date)
    end = pd.Timestamp(cfg.end_date)
    total_months = (end.year - start.year) * 12 + (end.month - start.month) + 1
    # Observation month index (0 = start_date month), roughly uniform across
    # the timeline so temporal train/validation/OOT splits all have volume.
    obs_month_idx = rng.integers(0, total_months, size=n)
    observation_date = start + pd.to_timedelta(obs_month_idx * 30, unit="D")
    observation_date = observation_date.to_period("M").to_timestamp("M")

    # Account tenure: mixture of new and seasoned accounts (gamma-shaped).
    account_tenure_months = np.clip(rng.gamma(shape=2.2, scale=10.0, size=n), 1, 180).round()
    origination_date = observation_date - pd.to_timedelta(account_tenure_months * 30, unit="D")
    origination_date = origination_date.to_period("M").to_timestamp("M")

    # Portfolio deterioration narrative: a slow upward drift in the latent
    # risk factor over calendar time, so later observation periods show
    # somewhat worse behavior on average (see docs/PORTFOLIO_STORY.md).
    time_drift = _zscore(obs_month_idx.astype(float)) * 0.18

    # --- Demographics (non-protected) ---------------------------------------
    age_band = rng.choice(AGE_BANDS, size=n, p=[0.14, 0.24, 0.24, 0.19, 0.13, 0.06])
    age_band_risk = (
        pd.Series(age_band)
        .map(
            {
                "18-24": 0.35,
                "25-34": 0.15,
                "35-44": 0.0,
                "45-54": -0.05,
                "55-64": -0.1,
                "65+": -0.05,
            }
        )
        .to_numpy()
    )

    employment_tenure_months = np.clip(rng.gamma(shape=2.0, scale=24.0, size=n), 0, 480)

    # --- Income & expenses ----------------------------------------------------
    monthly_income = np.clip(rng.lognormal(mean=8.15, sigma=0.55, size=n), 800, 90000)

    # Latent "financial discipline" factor drives several correlated fields.
    discipline = rng.normal(0, 1, size=n)

    expense_ratio = np.clip(0.55 - 0.08 * discipline + rng.normal(0, 0.08, size=n), 0.15, 0.95)
    declared_expenses = monthly_income * expense_ratio

    debt_ratio = np.clip(0.35 - 0.1 * discipline + rng.normal(0, 0.12, size=n), 0.0, 1.3)
    existing_debt = monthly_income * debt_ratio

    number_of_open_accounts = np.clip(
        rng.poisson(lam=np.clip(3 + discipline * 0.5, 0.5, None)), 0, 25
    )

    credit_limit = np.clip(
        monthly_income * rng.uniform(1.5, 4.5, size=n) * (1 - 0.15 * (discipline < -1)), 300, 200000
    )

    base_utilization = np.clip(0.45 - 0.15 * discipline + rng.normal(0, 0.18, size=n), 0.0, 1.4)
    revolving_balance = credit_limit * base_utilization
    credit_utilization = np.divide(
        revolving_balance,
        credit_limit,
        out=np.zeros_like(revolving_balance),
        where=credit_limit > 0,
    )

    # --- Delinquency history ---------------------------------------------------
    risk_propensity = (
        -0.9 * discipline
        + 0.6 * _zscore(base_utilization)
        + 0.5 * _zscore(debt_ratio)
        + age_band_risk
        + time_drift
        + rng.normal(0, 0.6, size=n)
    )
    risk_percentile = pd.Series(risk_propensity).rank(pct=True).to_numpy()

    late_lambda_12m = np.clip(0.15 + 3.5 * np.clip(risk_percentile - 0.5, 0, None) * 2, 0.02, 6)
    late_payments_12m = rng.poisson(lam=late_lambda_12m)
    late_payments_6m = np.minimum(late_payments_12m, rng.binomial(late_payments_12m, 0.65))
    late_payments_3m = np.minimum(late_payments_6m, rng.binomial(late_payments_6m, 0.55))

    max_days_past_due = np.where(
        late_payments_12m == 0,
        0,
        np.clip(rng.gamma(shape=1.3, scale=18 + 25 * risk_percentile, size=n), 0, 240),
    ).round()

    previous_default_flag = (
        rng.random(n) < np.clip(0.03 + 0.22 * risk_percentile**2, 0, 0.6)
    ).astype(int)

    payment_history_score = np.clip(1 - risk_percentile + rng.normal(0, 0.12, size=n), 0, 1)
    payment_history = pd.cut(
        payment_history_score,
        bins=[-0.01, 0.25, 0.5, 0.75, 1.01],
        labels=list(reversed(PAYMENT_HISTORY_LEVELS)),
    ).astype(str)

    # --- Ratios -----------------------------------------------------------------
    installment_to_income = np.clip(
        (existing_debt * 0.04) / np.maximum(monthly_income, 1) + rng.normal(0, 0.02, size=n),
        0,
        1.5,
    )
    debt_to_income = np.clip(
        existing_debt / np.maximum(monthly_income, 1) + installment_to_income * 0.3, 0, 3.0
    )

    average_monthly_spend = np.clip(
        monthly_income * rng.uniform(0.2, 0.6, size=n) * (1 - 0.1 * discipline), 0, None
    )
    cash_advance_frequency = np.clip(
        rng.poisson(lam=np.clip(0.3 + 2.0 * risk_percentile, 0.05, None)), 0, 20
    )
    number_of_recent_credit_inquiries = np.clip(
        rng.poisson(lam=np.clip(0.5 + 3.0 * risk_percentile, 0.1, None)), 0, 15
    )

    behavioral_score = np.clip(100 * (1 - risk_percentile) + rng.normal(0, 6, size=n), 0, 100)
    financial_stability_index = np.clip(
        0.5 + 0.3 * discipline - 0.2 * _zscore(debt_to_income) + rng.normal(0, 0.1, size=n),
        0,
        1,
    )
    transaction_volatility = np.clip(
        0.15 + 0.3 * risk_percentile + rng.normal(0, 0.08, size=n), 0.01, 1.5
    )

    balance_trend = np.clip(
        0.10 * risk_percentile - 0.05 * discipline + rng.normal(0, 0.08, size=n), -0.5, 1.0
    )
    utilization_trend = np.clip(0.08 * risk_percentile + rng.normal(0, 0.06, size=n), -0.4, 0.8)
    delinquency_trend = np.clip(0.12 * risk_percentile + rng.normal(0, 0.07, size=n), -0.4, 1.0)

    # --- Target: probability of 90+ day default within the next 90 days --------
    latent_score = (
        1.35 * _zscore(risk_propensity)
        + 0.9 * _zscore(debt_to_income)
        + 0.8 * _zscore(late_payments_12m.astype(float))
        + 0.7 * _zscore(credit_utilization)
        + 0.6 * previous_default_flag
        + 0.5 * _zscore(balance_trend)
        + 0.4 * _zscore(delinquency_trend)
        - 0.5 * _zscore(financial_stability_index)
        + rng.normal(0, 1.0, size=n)  # irreducible noise
    )
    # Calibrate intercept so the overall default rate matches the configured
    # realistic target (default ~7.5%) regardless of upstream distributional
    # tweaks to the latent risk factors.
    slope = 0.85
    intercept = _calibrate_intercept(latent_score, cfg.target_default_rate, slope=slope)
    default_probability = _sigmoid(intercept + slope * latent_score)
    target_default_90d = rng.binomial(1, default_probability)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "origination_date": origination_date,
            "observation_date": observation_date,
            "age_band": age_band,
            "monthly_income": monthly_income.round(2),
            "employment_tenure_months": employment_tenure_months.round(1),
            "declared_expenses": declared_expenses.round(2),
            "existing_debt": existing_debt.round(2),
            "number_of_open_accounts": number_of_open_accounts.astype(int),
            "credit_utilization": credit_utilization.round(4),
            "revolving_balance": revolving_balance.round(2),
            "payment_history": payment_history,
            "late_payments_3m": late_payments_3m.astype(int),
            "late_payments_6m": late_payments_6m.astype(int),
            "late_payments_12m": late_payments_12m.astype(int),
            "max_days_past_due": max_days_past_due.astype(int),
            "debt_to_income": debt_to_income.round(4),
            "installment_to_income": installment_to_income.round(4),
            "credit_limit": credit_limit.round(2),
            "average_monthly_spend": average_monthly_spend.round(2),
            "cash_advance_frequency": cash_advance_frequency.astype(int),
            "account_tenure_months": account_tenure_months.astype(int),
            "number_of_recent_credit_inquiries": number_of_recent_credit_inquiries.astype(int),
            "previous_default_flag": previous_default_flag.astype(int),
            "behavioral_score": behavioral_score.round(2),
            "financial_stability_index": financial_stability_index.round(4),
            "transaction_volatility": transaction_volatility.round(4),
            "balance_trend": balance_trend.round(4),
            "utilization_trend": utilization_trend.round(4),
            "delinquency_trend": delinquency_trend.round(4),
            "target_default_90d": target_default_90d.astype(int),
        }
    )

    # --- Realism: missing values, outliers, duplicate-ish noise -----------------
    for col, rate in [
        ("declared_expenses", 0.03),
        ("average_monthly_spend", 0.04),
        ("employment_tenure_months", 0.02),
        ("max_days_past_due", 0.015),
    ]:
        df[col] = _inject_missing(rng, df[col], rate)

    for col, rate, mult_range in [
        ("monthly_income", 0.004, (3.0, 6.0)),
        ("revolving_balance", 0.006, (1.5, 2.5)),
        ("credit_utilization", 0.005, (1.2, 1.8)),
    ]:
        df[col] = _inject_outliers(rng, df[col], rate, mult_range)

    logger.info(
        "generated_customer_snapshot",
        n_rows=len(df),
        default_rate=round(float(df["target_default_90d"].mean()), 4),
    )
    return df


def generate_monthly_performance_panel(
    customers: pd.DataFrame, cfg: GenerationConfig, max_mob: int = 18
) -> pd.DataFrame:
    """Generate a monthly delinquency-bucket panel per customer.

    Used downstream for vintage curves, months-on-book (MOB) analysis and
    roll-rate transition matrices. Simulated via a decile-conditioned Markov
    chain over `DELINQUENCY_BUCKETS`, seeded by each customer's relative risk
    percentile so riskier customers migrate to worse buckets more often and
    cure less often - without ever re-using the exact target label.
    """
    rng = np.random.default_rng(cfg.random_seed + 1)
    n = len(customers)
    logger.info("generating_monthly_performance_panel", n_customers=n, max_mob=max_mob)

    # Recompute a risk decile independently of the target draw (based on
    # observable behavioral fields only) so the panel is plausible but not a
    # deterministic function of the label.
    risk_proxy = (
        _zscore(customers["debt_to_income"].fillna(customers["debt_to_income"].median()))
        + _zscore(customers["credit_utilization"].fillna(customers["credit_utilization"].median()))
        + _zscore(customers["late_payments_12m"].astype(float))
        + _zscore(customers["delinquency_trend"])
    )
    decile = pd.qcut(risk_proxy, 10, labels=False, duplicates="drop")
    decile = decile.fillna(decile.median()).astype(int).to_numpy()

    n_states = len(DELINQUENCY_BUCKETS)

    # Build one transition matrix per decile (0=safest .. 9=riskiest).
    transition_matrices = []
    for d in range(10):
        risk_level = d / 9.0
        roll_forward = 0.02 + 0.10 * risk_level
        cure_rate = 0.55 - 0.35 * risk_level
        mat = np.zeros((n_states, n_states))
        for i in range(n_states):
            if i == 0:  # CURRENT
                mat[i, 0] = 1 - roll_forward
                mat[i, 1] = roll_forward
            elif i == n_states - 1:  # 90+ is absorbing-ish (charge-off track)
                mat[i, i] = 0.97
                mat[i, 0] = 0.03
            else:
                mat[i, min(i + 1, n_states - 1)] = roll_forward * 1.3
                mat[i, 0] = cure_rate
                mat[i, i] = max(1 - roll_forward * 1.3 - cure_rate, 0.01)
            mat[i] = mat[i] / mat[i].sum()
        transition_matrices.append(mat)
    transition_matrices = np.stack(transition_matrices)  # (10, 5, 5)

    tenure = customers["account_tenure_months"].to_numpy()
    origination = customers["origination_date"].to_numpy()
    mob_cap = np.minimum(tenure, max_mob).astype(int)

    current_state = np.zeros(n, dtype=int)  # everyone starts CURRENT at MOB 0
    balances = customers["revolving_balance"].fillna(0).to_numpy().astype(float)
    limits = customers["credit_limit"].to_numpy().astype(float)

    records = []
    max_horizon = int(mob_cap.max()) if n else 0
    for mob in range(0, max_horizon + 1):
        active = mob <= mob_cap
        if not active.any():
            continue
        idx = np.where(active)[0]
        if mob > 0:
            for d in range(10):
                sel = idx[decile[idx] == d]
                if len(sel) == 0:
                    continue
                mat = transition_matrices[d]
                for s in range(n_states):
                    s_sel = sel[current_state[sel] == s]
                    if len(s_sel) == 0:
                        continue
                    draws = rng.choice(n_states, size=len(s_sel), p=mat[s])
                    current_state[s_sel] = draws

        dpd_map = np.array([0, 15, 45, 75, 120])
        snapshot_month = (pd.Series(origination[idx]).dt.to_period("M") + mob).dt.to_timestamp("M")
        util_noise = rng.normal(0, 0.03, size=len(idx))
        state_now = current_state[idx]
        bucket_util_bump = state_now * 0.05
        util = np.clip(
            np.divide(balances[idx], limits[idx], out=np.zeros(len(idx)), where=limits[idx] > 0)
            + bucket_util_bump
            + util_noise,
            0,
            1.6,
        )
        records.append(
            pd.DataFrame(
                {
                    "customer_id": customers["customer_id"].to_numpy()[idx],
                    "snapshot_month": snapshot_month.to_numpy(),
                    "mob": mob,
                    "delinquency_bucket": np.array(DELINQUENCY_BUCKETS)[state_now],
                    "balance": balances[idx].round(2),
                    "utilization": util.round(4),
                    "days_past_due": dpd_map[state_now],
                }
            )
        )

    panel = pd.concat(records, ignore_index=True)
    logger.info("generated_monthly_performance_panel", n_rows=len(panel))
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic credit portfolio data.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-mob", type=int, default=18)
    args = parser.parse_args()

    configure_logging(get_settings().log_level)
    cfg = load_generation_config()

    output_dir = Path(args.output_dir) if args.output_dir else get_settings().data_path / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    customers = generate_customer_snapshot(cfg)
    panel = generate_monthly_performance_panel(customers, cfg, max_mob=args.max_mob)

    customers_path = output_dir / "customer_portfolio.parquet"
    panel_path = output_dir / "monthly_performance.parquet"
    customers.to_parquet(customers_path, index=False)
    panel.to_parquet(panel_path, index=False)

    logger.info(
        "synthetic_data_written",
        customers_path=str(customers_path.relative_to(PROJECT_ROOT)),
        panel_path=str(panel_path.relative_to(PROJECT_ROOT)),
        n_customers=len(customers),
        n_panel_rows=len(panel),
    )


if __name__ == "__main__":
    main()
