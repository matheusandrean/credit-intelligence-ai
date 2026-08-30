"""Data quality validation for the synthetic credit portfolio.

Combines a `pandera` schema (types, ranges, nullability) with a set of
business-rule checks (duplicates, impossible values, cross-field
consistency, distribution shifts) and emits a structured, JSON-serializable
report consumed by the Data Quality page in the Streamlit app and by CI.

This module never raises on a failed *business* check - it collects
failures into the report so a single bad batch does not crash the pipeline
silently. Structural failures (wrong dtype/missing required column) raise
`pandera.errors.SchemaError` because those indicate a broken upstream
contract that must be fixed, not merely monitored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from src.data.schema import AGE_BANDS, PAYMENT_HISTORY_LEVELS, PROHIBITED_PROTECTED_ATTRIBUTES
from src.utils.config import PROJECT_ROOT
from src.utils.logging import get_logger

logger = get_logger(__name__)

MAX_PLAUSIBLE_UTILIZATION = 2.0  # allow some over-limit spend, but bound it
MAX_PLAUSIBLE_DTI = 5.0
NULL_RATE_WARN_THRESHOLD = 0.10

customer_portfolio_schema = DataFrameSchema(
    {
        "customer_id": Column(str, Check.str_matches(r"^CUST_\d{6}$"), unique=True),
        "origination_date": Column("datetime64[ns]"),
        "observation_date": Column("datetime64[ns]"),
        "age_band": Column(str, Check.isin(AGE_BANDS)),
        "monthly_income": Column(float, Check.gt(0), nullable=True),
        "employment_tenure_months": Column(float, Check.ge(0), nullable=True),
        "declared_expenses": Column(float, Check.ge(0), nullable=True),
        "existing_debt": Column(float, Check.ge(0), nullable=True),
        "number_of_open_accounts": Column(int, Check.ge(0)),
        "credit_utilization": Column(
            float, Check.in_range(0, MAX_PLAUSIBLE_UTILIZATION), nullable=True
        ),
        "revolving_balance": Column(float, Check.ge(0), nullable=True),
        "payment_history": Column(str, Check.isin(PAYMENT_HISTORY_LEVELS)),
        "late_payments_3m": Column(int, Check.ge(0)),
        "late_payments_6m": Column(int, Check.ge(0)),
        "late_payments_12m": Column(int, Check.ge(0)),
        "max_days_past_due": Column(float, Check.ge(0), nullable=True),
        "debt_to_income": Column(float, Check.in_range(0, MAX_PLAUSIBLE_DTI), nullable=True),
        "installment_to_income": Column(float, Check.ge(0), nullable=True),
        "credit_limit": Column(float, Check.gt(0)),
        "average_monthly_spend": Column(float, Check.ge(0), nullable=True),
        "cash_advance_frequency": Column(int, Check.ge(0)),
        "account_tenure_months": Column(int, Check.ge(0)),
        "number_of_recent_credit_inquiries": Column(int, Check.ge(0)),
        "previous_default_flag": Column(int, Check.isin([0, 1])),
        "behavioral_score": Column(float, Check.in_range(0, 100)),
        "financial_stability_index": Column(float, Check.in_range(0, 1)),
        "transaction_volatility": Column(float, Check.ge(0)),
        "balance_trend": Column(float),
        "utilization_trend": Column(float),
        "delinquency_trend": Column(float),
        "target_default_90d": Column(int, Check.isin([0, 1])),
    },
    coerce=False,
    strict=False,
)


@dataclass
class DataQualityReport:
    generated_at: str
    n_rows: int
    n_columns: int
    schema_valid: bool
    schema_errors: list[str] = field(default_factory=list)
    null_rates: dict[str, float] = field(default_factory=dict)
    high_null_columns: list[str] = field(default_factory=list)
    duplicate_customer_ids: int = 0
    protected_attributes_found: list[str] = field(default_factory=list)
    consistency_failures: list[str] = field(default_factory=list)
    monthly_volume_anomalies: list[dict] = field(default_factory=list)
    default_rate_by_month: dict[str, float] = field(default_factory=dict)
    passed: bool = True

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "schema_valid": self.schema_valid,
            "schema_errors": self.schema_errors,
            "null_rates": self.null_rates,
            "high_null_columns": self.high_null_columns,
            "duplicate_customer_ids": self.duplicate_customer_ids,
            "protected_attributes_found": self.protected_attributes_found,
            "consistency_failures": self.consistency_failures,
            "monthly_volume_anomalies": self.monthly_volume_anomalies,
            "default_rate_by_month": self.default_rate_by_month,
            "passed": self.passed,
        }


def _check_consistency(df: pd.DataFrame) -> list[str]:
    failures = []
    if (df["late_payments_3m"] > df["late_payments_6m"]).any():
        failures.append("late_payments_3m must be <= late_payments_6m for all rows")
    if (df["late_payments_6m"] > df["late_payments_12m"]).any():
        failures.append("late_payments_6m must be <= late_payments_12m for all rows")
    if ((df["revolving_balance"] > 0) & (df["credit_limit"] <= 0)).any():
        failures.append("revolving_balance > 0 with non-positive credit_limit")
    if (df["observation_date"] < df["origination_date"]).any():
        failures.append("observation_date earlier than origination_date")
    return failures


def _check_monthly_anomalies(df: pd.DataFrame, z_threshold: float = 2.5) -> list[dict]:
    monthly = df.groupby(df["observation_date"].dt.to_period("M")).size()
    if len(monthly) < 3:
        return []
    z = (monthly - monthly.mean()) / monthly.std(ddof=0)
    anomalies = z[z.abs() > z_threshold]
    return [
        {"month": str(month), "count": int(monthly[month]), "z_score": round(float(val), 2)}
        for month, val in anomalies.items()
    ]


def run_data_quality_checks(df: pd.DataFrame) -> DataQualityReport:
    """Run the full data-quality suite and return a structured report."""
    logger.info("running_data_quality_checks", n_rows=len(df))

    schema_valid = True
    schema_errors: list[str] = []
    try:
        customer_portfolio_schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        schema_valid = False
        schema_errors = [
            f"{row['column']}: {row['check']}" for _, row in exc.failure_cases.iterrows()
        ][:50]

    null_rates = df.isna().mean().round(4).to_dict()
    high_null_columns = [c for c, r in null_rates.items() if r > NULL_RATE_WARN_THRESHOLD]

    duplicate_customer_ids = int(df["customer_id"].duplicated().sum())

    protected_found = [c for c in df.columns if c.lower() in PROHIBITED_PROTECTED_ATTRIBUTES]

    consistency_failures = _check_consistency(df)
    monthly_anomalies = _check_monthly_anomalies(df)

    default_rate_by_month = (
        df.groupby(df["observation_date"].dt.to_period("M"))["target_default_90d"].mean().round(4)
    )
    default_rate_by_month = {str(k): v for k, v in default_rate_by_month.items()}

    passed = (
        schema_valid
        and duplicate_customer_ids == 0
        and not protected_found
        and not consistency_failures
    )

    report = DataQualityReport(
        generated_at=datetime.now(UTC).isoformat(),
        n_rows=len(df),
        n_columns=len(df.columns),
        schema_valid=schema_valid,
        schema_errors=schema_errors,
        null_rates={k: float(v) for k, v in null_rates.items()},
        high_null_columns=high_null_columns,
        duplicate_customer_ids=duplicate_customer_ids,
        protected_attributes_found=protected_found,
        consistency_failures=consistency_failures,
        monthly_volume_anomalies=monthly_anomalies,
        default_rate_by_month=default_rate_by_month,
        passed=passed,
    )
    logger.info("data_quality_checks_complete", passed=passed, schema_valid=schema_valid)
    return report


def write_report(report: DataQualityReport, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "data_quality_report.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    md_lines = [
        "# Data Quality Report",
        "",
        f"Generated at: {report.generated_at}",
        "",
        f"- Rows: {report.n_rows:,}",
        f"- Columns: {report.n_columns}",
        f"- Schema valid: {report.schema_valid}",
        f"- Duplicate customer_id: {report.duplicate_customer_ids}",
        f"- Protected attributes found: {report.protected_attributes_found or 'None'}",
        f"- Overall passed: **{report.passed}**",
        "",
        "## Columns with null rate > " f"{NULL_RATE_WARN_THRESHOLD:.0%}",
        "",
    ]
    if report.high_null_columns:
        md_lines += [f"- {c}: {report.null_rates[c]:.2%}" for c in report.high_null_columns]
    else:
        md_lines.append("None")

    md_lines += ["", "## Consistency failures", ""]
    md_lines += [f"- {f}" for f in report.consistency_failures] or ["None"]

    md_lines += ["", "## Monthly volume anomalies (|z| > 2.5)", ""]
    md_lines += [
        f"- {a['month']}: count={a['count']} (z={a['z_score']})"
        for a in report.monthly_volume_anomalies
    ] or ["None"]

    md_path = output_dir / "data_quality_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path


def main() -> None:
    from src.utils.config import get_settings
    from src.utils.logging import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level)

    customers_path = settings.data_path / "raw" / "customer_portfolio.parquet"
    df = pd.read_parquet(customers_path)
    report = run_data_quality_checks(df)
    output_dir = PROJECT_ROOT / "reports"
    path = write_report(report, output_dir)
    logger.info("data_quality_report_written", path=str(path.relative_to(PROJECT_ROOT)))
    if not report.passed:
        logger.warning("data_quality_checks_failed", schema_errors=report.schema_errors[:5])


if __name__ == "__main__":
    main()
