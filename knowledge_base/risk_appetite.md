# Risk Appetite Statement

> **SYNTHETIC DEMONSTRATION POLICY.** Fictitious content for the Credit
> Intelligence AI portfolio project. Not a real institutional risk appetite
> statement.

## 1. Purpose

Defines illustrative portfolio-level risk tolerance thresholds used by the
Credit Intelligence AI demonstration platform's monitoring and stress-testing
modules.

## 2. Portfolio-Level Targets (Demonstrative)

| Indicator | Target / Tolerance | Escalation Trigger |
|---|---|---|
| Portfolio average PD | Monitor if > 10% | Escalate if > 15% |
| Expected Loss rate (EL / EAD) | Monitor if > 6% | Escalate if > 9% |
| High-risk population (Bands D+E) | Monitor if > 20% | Escalate if > 30% |
| 90+ delinquency rate | Monitor if > 3% | Escalate if > 5% |

These thresholds are illustrative reference points for the demonstration
dashboard only. They are not calibrated to any real institution's capital,
funding cost, or regulatory position.

## 3. Risk Band Usage

Risk bands A-E, produced by converting model PD into an illustrative 300-900
score (see `MODEL_CARD.md` and `FEATURE_DICTIONARY.md`), are used strictly
for internal monitoring and reporting segmentation in this demonstration.
They are explicitly NOT a commercial pricing or approval policy.

## 4. Stress Testing Appetite

The platform's stress-testing module (see `run_stress_test` tool and
`docs/STRESS_TESTING.md`) evaluates portfolio resilience under Mild,
Moderate and Severe scenarios. A Severe scenario producing a portfolio
average PD increase of more than 3 percentage points relative to baseline
is treated, for demonstration purposes, as a signal warranting escalation
to a Credit Risk Director-level review.

## 5. Concentration Considerations

The demonstration platform does not currently model geographic, industry or
single-obligor concentration limits. This is a known and documented scope
limitation (see `docs/LIMITATIONS.md`), not an oversight to be inferred as
resolved.

## 6. Relationship to Model Risk

Risk appetite thresholds here are read in conjunction with model
performance monitoring (PSI, AUC/KS drift) described in
`model_governance_policy.md`. A materially drifted model can invalidate the
risk appetite comparisons above even if the raw indicators look stable.
