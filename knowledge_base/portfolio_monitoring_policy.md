# Portfolio Monitoring Policy

> **SYNTHETIC DEMONSTRATION POLICY.** Fictitious content for the Credit
> Intelligence AI portfolio project. Not a real institutional policy.

## 1. Purpose

Describes the illustrative cadence and content of portfolio monitoring
activities supported by the platform's analytics modules (portfolio KPIs,
vintage/MOB analysis, roll-rate analysis, and drift monitoring).

## 2. Monthly Monitoring Package (Demonstrative)

Each simulated monitoring cycle reviews:

1. **Portfolio KPIs**: exposure, average PD, Expected Loss, default rate,
   delinquency rate, high-risk population %, average DTI and utilization,
   month-over-month (MoM) and quarter-over-quarter (QoQ) deltas.
2. **Vintage / Months-on-Book (MOB) analysis**: bad-rate curves by
   origination cohort, to distinguish genuine credit-quality deterioration
   from simple portfolio growth or seasoning effects.
3. **Roll-rate analysis**: month-over-month migration between delinquency
   buckets (CURRENT, 1-30, 31-60, 61-90, 90+), including cure rates.
4. **Model performance and drift**: ROC-AUC, KS, Gini over time, and
   Population Stability Index (PSI) for both model inputs and the predicted
   PD distribution (see `model_governance_policy.md`).

## 3. Deterioration Watchlist

An account or cohort is added to the demonstrative deterioration watchlist
when any of the following hold in a given monitoring cycle:

- `delinquency_trend` and `utilization_trend` are both positive and above
  the 75th percentile of the current portfolio distribution.
- A vintage cohort's bad rate at a given MOB checkpoint exceeds the same
  MOB checkpoint bad rate of the prior three cohorts by more than 50%
  relative.
- Roll-rate from CURRENT to 1-30+ increases materially versus the trailing
  3-month average (see `run_stress_test` and `detect_drift` tool outputs).

## 4. Reporting

Findings are summarized in an Executive Portfolio Report (see
`docs/EXECUTIVE_REPORTING.md`) covering: portfolio overview, risk movement,
main drivers (via SHAP), deterioration signals, stress-testing results,
model health, and points requiring human attention. The report is always
data-driven; narrative commentary must cite the specific tool output or
retrieved policy section it is based on.

## 5. Escalation

Any indicator crossing an Escalation Trigger defined in
`risk_appetite.md` is routed to human review. No automated action is taken
by the platform as a result of a monitoring finding.
