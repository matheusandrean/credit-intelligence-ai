# Credit Policy

> **SYNTHETIC DEMONSTRATION POLICY.** This document is fictitious content
> created for the Credit Intelligence AI portfolio project. It does not
> represent the policy of any real financial institution and must never be
> treated as legal, regulatory or commercial guidance.

## 1. Purpose

This policy defines the general principles governing unsecured revolving
consumer credit lines originated and monitored through the Credit
Intelligence AI demonstration platform.

## 2. Scope

Applies to all synthetic retail customers represented in the platform's
demonstration portfolio. Does not apply to secured lending, commercial
lending, or any real customer data, none of which exist in this project.

## 3. Eligibility Criteria (Demonstrative)

A synthetic account is considered eligible for credit line origination when
all of the following illustrative conditions are met:

- Debt-to-Income (DTI) ratio below 0.65.
- No confirmed default in the previous 12 months (`previous_default_flag = 0`),
  OR a documented, human-reviewed exception.
- Account tenure and behavioral history sufficient to compute a Probability
  of Default (PD) estimate, or eligibility for thin-file treatment (Section 6).

## 4. High Debt-Service Commitment Segment

Customers with **elevated debt-service commitment** — defined for this
demonstration as `debt_to_income >= 0.50` OR `installment_to_income >= 0.35`
— fall under enhanced monitoring:

- Credit line increases are not automatically recommended by any quantitative
  tool; any change requires human credit-analyst review.
- These accounts are prioritized in the monthly Portfolio Monitoring review
  (see `portfolio_monitoring_policy.md`).
- The AI Credit Analyst assistant may surface these customers in response to
  queries such as "clientes com comprometimento elevado" but must present
  them as candidates for analyst review, not as automatic decisions.

## 5. Utilization Thresholds (Demonstrative)

- Utilization above 0.75 is treated as an early-warning behavioral signal.
- Utilization above 0.90 combined with an increasing `utilization_trend`
  triggers inclusion in the deterioration watchlist described in
  `portfolio_monitoring_policy.md`.

## 6. Thin-File Segment

Accounts with `account_tenure_months <= 6` and `number_of_open_accounts <= 1`
are classified as "thin file." Quantitative PD estimates for this segment
carry higher uncertainty and should be interpreted with additional caution
by the reviewing analyst (see `model_governance_policy.md`, Section 5, on
reject inference and population coverage limits).

## 7. Human Oversight

No component of this platform is authorized to approve, deny, or modify a
real credit decision. Every output is decision SUPPORT for a human credit
analyst or risk manager. This is a structural requirement of the platform,
not merely a configuration option (see `responsible_ai_policy.md`).

## 8. Review Cadence

This demonstrative policy is reviewed conceptually alongside each model
retraining cycle described in `model_governance_policy.md`.
