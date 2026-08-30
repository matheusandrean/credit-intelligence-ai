"""Train and evaluate the credit risk models.

Trains two models on the temporally-split feature dataset:

1. Logistic Regression - the interpretable baseline used for governance
   discussions and as a sanity-check champion.
2. LightGBM - the challenger, expected to capture non-linear interactions
   at some cost to direct interpretability (mitigated with SHAP).

Both models are evaluated on train / validation / test(OOT) with the same
metric suite (never accuracy alone - see src/models/metrics.py) and logged
to MLflow together with the dataset split boundaries and feature list, so
every run is reproducible and auditable.
"""

from __future__ import annotations

import json
import os
from typing import Any

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.feature_lists import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    MODEL_NUMERIC_FEATURES,
)
from src.models.metrics import ClassificationMetrics, compute_classification_metrics
from src.utils.config import PROJECT_ROOT, get_settings
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "target_default_90d"
RANDOM_SEED = 42


def load_dataset() -> pd.DataFrame:
    settings = get_settings()
    path = settings.data_path / "processed" / "credit_features.parquet"
    return pd.read_parquet(path)


def _cast_categoricals(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    for col, cats in categories.items():
        out[col] = pd.Categorical(out[col], categories=cats)
    return out


def split_xy(
    df: pd.DataFrame, split_name: str, categories: dict[str, list[str]]
) -> tuple[pd.DataFrame, np.ndarray]:
    subset = df[df["split"] == split_name]
    x = subset[list(MODEL_FEATURES)].copy()
    x = _cast_categoricals(x, categories)
    y = subset[TARGET_COLUMN].to_numpy()
    return x, y


def build_logistic_regression_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, list(MODEL_NUMERIC_FEATURES)),
            ("categorical", categorical_transformer, list(MODEL_CATEGORICAL_FEATURES)),
        ]
    )
    model = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED, C=0.5
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def train_logistic_regression(x_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    pipeline = build_logistic_regression_pipeline()
    pipeline.fit(x_train, y_train)
    return pipeline


def train_lightgbm(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="auc",
        categorical_feature=list(MODEL_CATEGORICAL_FEATURES),
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    return model


def evaluate_on_splits(
    predict_fn: Any, df: pd.DataFrame, categories: dict[str, list[str]]
) -> dict[str, ClassificationMetrics]:
    results = {}
    for split_name in ["train", "validation", "test"]:
        x, y = split_xy(df, split_name, categories)
        if len(x) == 0:
            continue
        y_prob = predict_fn(x)
        results[split_name] = compute_classification_metrics(y, y_prob)
    return results


def _metrics_summary(results: dict[str, ClassificationMetrics]) -> dict:
    return {split: m.to_dict() for split, m in results.items()}


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    with open(PROJECT_ROOT / "configs" / "project_config.yaml", encoding="utf-8") as f:
        project_cfg = yaml.safe_load(f)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(project_cfg["mlflow"]["experiment_name"])

    df = load_dataset()
    categories = {
        col: sorted(df[col].dropna().unique().tolist()) for col in MODEL_CATEGORICAL_FEATURES
    }

    x_train, y_train = split_xy(df, "train", categories)
    x_val, y_val = split_xy(df, "validation", categories)

    models_dir = settings.models_path
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparison: dict[str, dict] = {}

    # --- Model 1: Logistic Regression baseline -------------------------------
    with mlflow.start_run(run_name="logistic_regression_baseline"):
        logger.info("training_logistic_regression", n_train=len(x_train))
        lr_pipeline = train_logistic_regression(x_train, y_train)
        lr_results = evaluate_on_splits(
            lambda x: lr_pipeline.predict_proba(x)[:, 1], df, categories
        )
        mlflow.log_params(
            {"model_type": "logistic_regression", "C": 0.5, "class_weight": "balanced"}
        )
        for split_name, m in lr_results.items():
            mlflow.log_metrics(
                {
                    f"{split_name}_{k}": v
                    for k, v in m.to_dict().items()
                    if isinstance(v, int | float)
                }
            )
        mlflow.sklearn.log_model(lr_pipeline, name="model", serialization_format="pickle")
        joblib.dump(lr_pipeline, models_dir / "logistic_regression.joblib")
        comparison["logistic_regression"] = _metrics_summary(lr_results)
        logger.info(
            "logistic_regression_test_auc",
            auc=lr_results.get("test", None) and lr_results["test"].roc_auc,
        )

    # --- Model 2: LightGBM challenger -----------------------------------------
    with mlflow.start_run(run_name="lightgbm_challenger"):
        logger.info("training_lightgbm", n_train=len(x_train))
        lgb_model = train_lightgbm(x_train, y_train, x_val, y_val)
        lgb_results = evaluate_on_splits(lambda x: lgb_model.predict_proba(x)[:, 1], df, categories)
        mlflow.log_params(
            {
                "model_type": "lightgbm",
                "n_estimators": lgb_model.n_estimators,
                "learning_rate": lgb_model.learning_rate,
                "num_leaves": lgb_model.num_leaves,
                "best_iteration": lgb_model.best_iteration_,
            }
        )
        for split_name, m in lgb_results.items():
            mlflow.log_metrics(
                {
                    f"{split_name}_{k}": v
                    for k, v in m.to_dict().items()
                    if isinstance(v, int | float)
                }
            )
        mlflow.lightgbm.log_model(lgb_model, name="model")
        joblib.dump(lgb_model, models_dir / "lightgbm.joblib")
        comparison["lightgbm"] = _metrics_summary(lgb_results)
        logger.info(
            "lightgbm_test_auc", auc=lgb_results.get("test", None) and lgb_results["test"].roc_auc
        )

    # Champion selection: prefer the model with higher OOT (test) ROC-AUC,
    # documented explicitly rather than silently assumed.
    champion = max(
        comparison.keys(), key=lambda name: comparison[name].get("test", {}).get("roc_auc", 0)
    )

    metadata = {
        "feature_columns": {
            "numeric": list(MODEL_NUMERIC_FEATURES),
            "categorical": list(MODEL_CATEGORICAL_FEATURES),
        },
        "categories": categories,
        "champion_model": champion,
        "random_seed": RANDOM_SEED,
        "n_train": int(len(x_train)),
        "n_validation": int(len(x_val)),
    }
    (models_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (reports_dir / "model_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    logger.info("training_complete", champion_model=champion)


if __name__ == "__main__":
    main()
