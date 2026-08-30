# Model Governance Policy

> **SYNTHETIC DEMONSTRATION POLICY.** Fictitious content for the Credit
> Intelligence AI portfolio project. Not a real institutional Model Risk
> Management (MRM) framework.

## 1. Purpose

Describes the illustrative governance lifecycle applied to the credit risk
models (Logistic Regression baseline, LightGBM challenger) within this
demonstration platform.

## 2. Model Development Standards

- Models are trained and evaluated using strict temporal (out-of-time)
  validation: train / validation / test windows are separated by calendar
  date, never by random shuffling, to avoid look-ahead bias
  (`src/features/build_features.py`, `docs/TEMPORAL_VALIDATION.md`).
- Evaluation always reports discrimination (ROC-AUC, KS, Gini), calibration
  (Brier score, reliability curve) and business metrics (lift, recall at
  top deciles) - never accuracy alone, which is uninformative for an
  imbalanced default-prediction problem.
- Champion selection is based on out-of-time test ROC-AUC and is recorded,
  with the full comparison, in `models/model_metadata.json` and
  `reports/model_comparison.json`.

## 3. Calibration Standard

Raw classifier probabilities are calibrated (Platt scaling or isotonic
regression, selected by validation-set Brier score) before being reported
as "Probability of Default." Calibration is fit on the validation split
only, never on training data, to avoid overfitting the calibration itself.

## 4. Monitoring Thresholds (Demonstrative)

Population Stability Index (PSI) thresholds used by `detect_drift`:

- PSI < 0.10: Stable.
- 0.10 <= PSI < 0.25: Monitor.
- PSI >= 0.25: Potential significant drift, escalate for review.

These thresholds are industry-common conventions adapted for this
demonstration; a real deployment must calibrate them to its own governance
framework and validate them independently.

## 5. Known Limitations (Reject Inference)

This demonstration platform trains only on accounts that were originated
(there is no "declined applicant" population). This is the classic reject
inference problem in credit scoring: a model trained only on approved
accounts can be biased relative to the true through-the-door population.
This platform does NOT implement or claim to implement reject inference; it
is discussed here only as a conceptual limitation for transparency, per
`docs/LIMITATIONS.md`.

## 6. Retraining and Change Management

In this demonstration, retraining is triggered conceptually by: (a)
material drift detected by `detect_drift`, (b) a full new data-generation
cycle, or (c) a documented change to the feature pipeline. Any such change
must re-run the full evaluation suite before a new model is considered for
"champion" status.

## 7. Independent Validation

This platform's models have NOT undergone independent validation, legal
review, or regulatory approval of any kind. They exist solely for portfolio
demonstration purposes (see repository `LICENSE` and `RESPONSIBLE_AI.md`).
