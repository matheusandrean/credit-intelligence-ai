# Temporal (Out-of-Time) Validation

## Why not `train_test_split(random_state=42)`

A random split shuffles rows without regard to calendar time. For a credit
model, this creates **look-ahead bias**: the model can implicitly learn
information about how the future portfolio behaves (macro conditions,
seasonal effects, mix shift) because "future" rows leak into training data
that sit right next to "past" rows the model is evaluated on. A model that
looks good under a random split can perform much worse in real production,
where it only ever sees the future.

## What this project does instead

`src/features/build_features.py::assign_temporal_split` partitions every
row strictly by `observation_date` against fixed calendar boundaries in
`configs/project_config.yaml`:

| Split | Window | Rows (last generation run) |
|---|---|---:|
| Train | 2023-01-01 to 2024-06-30 | 63,234 |
| Validation | 2024-07-01 to 2024-12-31 | 20,180 |
| Test (OOT) | 2025-01-01 to 2025-06-30 | 16,586 |

The test set genuinely represents a later period than train/validation —
enforced and tested in
`tests/unit/test_build_features.py::test_build_feature_dataset_adds_split_column`,
which asserts `train.observation_date.max() <= test.observation_date.min()`.

## Leakage discussion

Beyond the split itself, two more leakage risks were specifically
addressed:

1. **Engineered features never use the target.** Every feature in
   `engineer_features()` is built only from raw, observable fields — see
   `tests/unit/test_build_features.py::test_engineered_features_do_not_use_target`.
2. **Calibration is fit on validation, not train.** Fitting Platt/isotonic
   calibration on the same data used to train the underlying classifier
   would overfit the calibration curve; `src/models/calibration.py` fits
   exclusively on the validation split and evaluates on the untouched OOT
   test split.

## Stability and generalization

Reporting metrics on all three splits (`reports/model_comparison.json`)
lets a reviewer see whether performance degrades from train → validation →
test. A large train-to-test AUC drop would signal overfitting or temporal
instability; in this project's last run, OOT test AUC (0.8357) is close to
validation AUC (0.8445) and train AUC (0.8436), consistent with a
well-generalizing model rather than one memorizing the training window.
