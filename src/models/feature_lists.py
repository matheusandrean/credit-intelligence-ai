"""Canonical feature lists used by training, scoring, SHAP and the API.

Kept separate from `src.data.schema` because the *model* feature list is a
subset of the full table (identifiers, dates and the split label are
excluded) and also includes engineered features from
`src.features.build_features`.
"""

from __future__ import annotations

from src.data.schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.features.build_features import ENGINEERED_FEATURE_COLUMNS

NON_FEATURE_COLUMNS: tuple[str, ...] = (
    "customer_id",
    "origination_date",
    "observation_date",
    "target_default_90d",
    "split",
)

MODEL_CATEGORICAL_FEATURES: tuple[str, ...] = CATEGORICAL_COLUMNS
MODEL_NUMERIC_FEATURES: tuple[str, ...] = NUMERIC_COLUMNS + ENGINEERED_FEATURE_COLUMNS
MODEL_FEATURES: tuple[str, ...] = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES
