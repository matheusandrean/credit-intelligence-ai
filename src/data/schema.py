"""Column names, dtypes and governance constants for the synthetic credit dataset.

This module is the single source of truth for column names so that the
generator, validation layer, feature pipeline and models never hardcode
strings independently.
"""

from __future__ import annotations

# Explicitly-banned protected/sensitive characteristics.
# The synthetic generator MUST NOT create these columns, and the feature
# pipeline / model training code MUST reject any dataset that contains them.
# See RESPONSIBLE_AI.md for the full policy.
PROHIBITED_PROTECTED_ATTRIBUTES: tuple[str, ...] = (
    "race",
    "color",
    "ethnicity",
    "religion",
    "sexual_orientation",
    "gender_identity",
    "gender",
    "sex",
    "political_opinion",
    "disability",
    "nationality",
    "marital_status",
)

CUSTOMER_ID = "customer_id"
TARGET = "target_default_90d"

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "age_band",
    "payment_history",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "monthly_income",
    "employment_tenure_months",
    "declared_expenses",
    "existing_debt",
    "number_of_open_accounts",
    "credit_utilization",
    "revolving_balance",
    "late_payments_3m",
    "late_payments_6m",
    "late_payments_12m",
    "max_days_past_due",
    "debt_to_income",
    "installment_to_income",
    "credit_limit",
    "average_monthly_spend",
    "cash_advance_frequency",
    "account_tenure_months",
    "number_of_recent_credit_inquiries",
    "previous_default_flag",
    "behavioral_score",
    "financial_stability_index",
    "transaction_volatility",
    "balance_trend",
    "utilization_trend",
    "delinquency_trend",
)

DATE_COLUMNS: tuple[str, ...] = ("origination_date", "observation_date")

CUSTOMER_TABLE_COLUMNS: tuple[str, ...] = (
    (CUSTOMER_ID,) + DATE_COLUMNS + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + (TARGET,)
)

DELINQUENCY_BUCKETS: tuple[str, ...] = ("CURRENT", "1-30", "31-60", "61-90", "90+")

AGE_BANDS: tuple[str, ...] = ("18-24", "25-34", "35-44", "45-54", "55-64", "65+")

PAYMENT_HISTORY_LEVELS: tuple[str, ...] = ("Excellent", "Good", "Fair", "Poor")

MONTHLY_PERFORMANCE_COLUMNS: tuple[str, ...] = (
    CUSTOMER_ID,
    "snapshot_month",
    "mob",
    "delinquency_bucket",
    "balance",
    "utilization",
    "days_past_due",
)
