# The Portfolio Story

This project is designed to tell one coherent narrative end to end, not
just showcase disconnected components.

> The synthetic portfolio grows across 2023-2025.
>
> A slow, deliberate upward drift in the underlying latent risk factor
> makes later-period customers mildly riskier on average
> (`src/data/generate_synthetic_credit_data.py`, `time_drift` term).
>
> Quantitative models (Logistic Regression + LightGBM) identify a
> realistic, out-of-time-validated Probability of Default for every
> account — OOT ROC-AUC ≈0.836, KS ≈0.53, deliberately tuned to a
> realistic band rather than an implausible ~0.95+ (see
> [`MODEL_CARD.md`](../MODEL_CARD.md)).
>
> Population Stability Index monitoring (`detect_drift`) tracks whether
> the input distributions and the predicted-PD distribution are shifting
> month over month.
>
> Vintage analysis (bad rate by origination cohort and months-on-book) and
> roll-rate analysis (delinquency-bucket migration) let an analyst localize
> *where* in the book any deterioration signal is concentrated — a cohort
> effect, an aging effect, or a broad-based shift.
>
> Explainable AI (SHAP) reveals, for any single customer or globally
> across the book, which factors are actually driving risk up or down —
> `debt_to_income`, `behavioral_score`, `late_payments_12m`, and
> `utilization_gap_to_limit` are consistently the top global drivers.
>
> Stress testing quantifies vulnerability under Mild/Moderate/Severe
> macro-shock scenarios: baseline average PD (≈7.5%) rises to ≈7.8% (mild),
> ≈8.7% (moderate), and ≈10.3% (severe) — see
> `reports/stress_test_report.json`.
>
> The AI Credit Analyst lets a human investigate all of the above in
> natural language — portfolio summary, segment comparison, customer
> drivers, drift indicators, stress scenarios, and policy citations —
> without ever approving, denying, or overriding a single decision.

Every number quoted above is measured, not invented — see
`reports/model_comparison.json`, `reports/calibration_report.json`,
`reports/stress_test_report.json`, and `reports/sql/*.csv`, all produced by
running this project's own pipeline.
