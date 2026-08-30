# Feature Dictionary

This document describes every field in the synthetic credit portfolio,
from the raw generated attributes through the engineered risk features used
by the models. All data is synthetic — see [`DATA_CARD.md`](DATA_CARD.md).

## 1. Identifiers and dates

| Field | Type | Description |
|---|---|---|
| `customer_id` | string | Synthetic sequential ID (`CUST_000001`, ...). Never a real identifier. |
| `origination_date` | date | Month the synthetic account was opened. |
| `observation_date` | date | Month of the snapshot used for modeling; drives the temporal train/validation/test split (see [`docs/TEMPORAL_VALIDATION.md`](docs/TEMPORAL_VALIDATION.md)). |

## 2. Raw customer attributes

| Field | Type | Description | Notes |
|---|---|---|---|
| `age_band` | categorical | One of `18-24, 25-34, 35-44, 45-54, 55-64, 65+`. | Not a protected characteristic in this project's scope; used as a coarse, bucketed demographic segment only. |
| `monthly_income` | float | Synthetic gross monthly income. | Log-normal distribution; ~0.4% outliers injected. |
| `employment_tenure_months` | float | Months in current employment. | ~2% missing (injected). |
| `declared_expenses` | float | Self-declared monthly expenses. | ~3% missing (injected). |
| `existing_debt` | float | Total existing monthly debt service. | |
| `number_of_open_accounts` | int | Count of other open credit accounts. | |
| `credit_utilization` | float | `revolving_balance / credit_limit`. | ~0.5% outliers (>100% utilization, over-limit spend). |
| `revolving_balance` | float | Outstanding revolving balance. | |
| `payment_history` | categorical | `Excellent, Good, Fair, Poor`, derived from a latent payment-behavior score. | |
| `late_payments_3m` / `_6m` / `_12m` | int | Count of late payments in the trailing window. | Monotonically consistent (`3m <= 6m <= 12m`), enforced and validated. |
| `max_days_past_due` | float | Worst delinquency in the observation window. | ~1.5% missing (injected). |
| `debt_to_income` (DTI) | float | `total_debt_payment / monthly_income`. | Primary risk driver. |
| `installment_to_income` | float | Installment payment share of income. | |
| `credit_limit` | float | Revolving credit limit. | |
| `average_monthly_spend` | float | Average monthly spend on the line. | ~4% missing (injected). |
| `cash_advance_frequency` | int | Cash-advance transactions in the observation window. | |
| `account_tenure_months` | int | Months since origination. | |
| `number_of_recent_credit_inquiries` | int | Recent credit-seeking behavior. | |
| `previous_default_flag` | 0/1 | Prior confirmed default. | Strong risk driver. |
| `behavioral_score` | float (0-100) | Internal behavioral score (higher = lower risk). | |
| `financial_stability_index` | float (0-1) | Composite stability indicator. | |
| `transaction_volatility` | float | Spend volatility proxy. | |
| `balance_trend` | float | Recent trend of revolving balance. | |
| `utilization_trend` | float | Recent trend of utilization. | |
| `delinquency_trend` | float | Recent trend of delinquency severity. | |

**Never present, by design:** race, color, ethnicity, religion, sexual
orientation, gender identity, political opinion, disability, nationality,
marital status — see `src/data/schema.py::PROHIBITED_PROTECTED_ATTRIBUTES`
and [`RESPONSIBLE_AI.md`](RESPONSIBLE_AI.md).

## 3. Engineered features (`src/features/build_features.py`)

Built exclusively from the raw fields above — never from the target, so
there is no label leakage.

| Feature | Formula | Rationale |
|---|---|---|
| `payment_stress_index` | `0.4*min(late_12m/6,1) + 0.35*min(DTI/1.5,1) + 0.25*min(utilization,1)` | Single bounded indicator combining delinquency, indebtedness and utilization. |
| `behavioral_deterioration_index` | `0.4*delinquency_trend + 0.35*utilization_trend + 0.25*balance_trend` | Aggregates the three directional trend signals into one deterioration score. |
| `recent_inquiry_intensity` | `recent_inquiries / max(account_tenure_months, 1)` | Normalizes credit-seeking behavior by account age. |
| `spend_to_income` | `average_monthly_spend / monthly_income` | Affordability proxy. |
| `utilization_gap_to_limit` | `1 - min(credit_utilization, 1)` | Remaining headroom on the line. |
| `is_thin_file` | `tenure <= 6 AND open_accounts <= 1` | Flags a segment that typically needs different underwriting treatment. |

## 4. Target

| Field | Type | Description |
|---|---|---|
| `target_default_90d` | 0/1 | 90-day default indicator. Generated from a latent risk score built from DTI, utilization, late payments, previous default, balance/delinquency trend and financial stability, plus substantial irreducible noise (calibrated so OOT model AUC lands in a realistic ~0.83-0.85 band, not an implausible ~0.95+ — see `src/data/generate_synthetic_credit_data.py`). Portfolio-wide default rate ≈ 7.5% by construction. |

## 5. Model outputs (`src/risk/scorebook.py`)

| Field | Description |
|---|---|
| `pd` | Calibrated Probability of Default from the champion model. |
| `score` | Illustrative 300-900 score via points-to-double-odds scaling. **Demonstrative only.** |
| `risk_band` | A-E band derived from `score`. **Demonstrative only** — not a real commercial policy. |
| `lgd` | Synthetic Loss Given Default (base rate + noise). |
| `ead` | Exposure at Default: drawn balance + 50% credit-conversion factor × undrawn commitment. |
| `expected_loss` | `PD × LGD × EAD`. |

## 6. Monthly performance panel (`data/raw/monthly_performance.parquet`)

One row per customer-month, used for vintage/MOB and roll-rate analytics.

| Field | Description |
|---|---|
| `snapshot_month` | Calendar month of this panel row. |
| `mob` | Months on book at this row (0 = origination month). |
| `delinquency_bucket` | `CURRENT, 1-30, 31-60, 61-90, 90+`, simulated via a decile-conditioned Markov chain (not literally re-derived from `target_default_90d`, but built from the same observable risk drivers). |
| `balance`, `utilization`, `days_past_due` | Point-in-time account state. |
