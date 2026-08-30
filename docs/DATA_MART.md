# Data Mart — Dimensional Model

Conceptual star schema over the synthetic portfolio, implemented as
DuckDB views (`sql/00_views.sql`) over the generated parquet files rather
than a physically materialized warehouse — appropriate for this
project's scale and the "everything reproducible from source" principle.

```mermaid
erDiagram
    DIM_CUSTOMER ||--o{ FACT_CREDIT_PERFORMANCE : "scored in"
    DIM_CUSTOMER ||--o{ FACT_CUSTOMER_BEHAVIOR : "observed in"
    DIM_CUSTOMER ||--o{ FACT_DELINQUENCY : "tracked monthly in"
    DIM_DATE ||--o{ FACT_CREDIT_PERFORMANCE : "observation_date"
    DIM_DATE ||--o{ FACT_DELINQUENCY : "snapshot_month"
    DIM_RISK_BAND ||--o{ FACT_CREDIT_PERFORMANCE : "risk_band"

    DIM_CUSTOMER {
        string customer_id PK
        date origination_date
        string age_band
    }
    DIM_DATE {
        date calendar_date PK
        int year
        int month
        string quarter
    }
    DIM_RISK_BAND {
        string band PK
        int min_score
        int max_score
        string description
    }
    FACT_CREDIT_PERFORMANCE {
        string customer_id FK
        date observation_date FK
        float pd
        float score
        string risk_band FK
        float lgd
        float ead
        float expected_loss
        int target_default_90d
    }
    FACT_CUSTOMER_BEHAVIOR {
        string customer_id FK
        float debt_to_income
        float credit_utilization
        int late_payments_12m
        float payment_stress_index
        float behavioral_deterioration_index
    }
    FACT_DELINQUENCY {
        string customer_id FK
        date snapshot_month FK
        int mob
        string delinquency_bucket
        float balance
        float utilization
        int days_past_due
    }
```

## Mapping to physical files

| Logical entity | Physical source |
|---|---|
| `fact_credit_performance` | `data/processed/scored_portfolio.parquet` (view `v_portfolio`) |
| `fact_customer_behavior` | `data/processed/credit_features.parquet` (view `v_features`) |
| `fact_delinquency` | `data/raw/monthly_performance.parquet` (view `v_monthly_performance`) |
| `dim_customer` | `data/raw/customer_portfolio.parquet` (view `v_customers`), keyed by `customer_id` |
| `dim_date` | Implicit — `observation_date`/`snapshot_month` columns, no separate materialized table |
| `dim_risk_band` | `configs/project_config.yaml::risk_bands` |

See `sql/*.sql` for report queries against this model, and
`src/analytics/run_sql_reports.py` for the runner that executes them
against real data and writes `reports/sql/*.csv`.

