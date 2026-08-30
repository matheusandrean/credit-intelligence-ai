"""Credit-risk model evaluation metrics.

Deliberately goes beyond accuracy (per project convention: accuracy is a
poor metric for imbalanced default prediction). Provides discrimination
(ROC-AUC, PR-AUC, KS, Gini), calibration (Brier score, reliability curve)
and business-relevant metrics (lift, recall at top deciles).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ClassificationMetrics:
    roc_auc: float
    pr_auc: float
    ks_statistic: float
    gini: float
    brier_score: float
    precision_at_50: float
    recall_at_50: float
    f1_at_50: float
    lift_at_top_decile: float
    recall_at_top_decile: float
    recall_at_top_2_deciles: float
    n_samples: int
    default_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation between the cumulative
    distributions of predicted scores for defaulters vs non-defaulters."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(np.abs(tpr - fpr)))


def lift_and_recall_at_top_k_pct(
    y_true: np.ndarray, y_prob: np.ndarray, pct: float
) -> tuple[float, float]:
    """Lift and recall when flagging the top `pct` fraction of scores as risk."""
    n = len(y_true)
    k = max(1, int(np.ceil(n * pct)))
    order = np.argsort(-y_prob)
    top_k_idx = order[:k]
    base_rate = y_true.mean()
    if base_rate == 0:
        return 0.0, 0.0
    top_k_rate = y_true[top_k_idx].mean()
    lift = float(top_k_rate / base_rate)
    recall = float(y_true[top_k_idx].sum() / y_true.sum()) if y_true.sum() > 0 else 0.0
    return lift, recall


def compute_classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> ClassificationMetrics:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    ks = ks_statistic(y_true, y_prob)
    gini = 2 * roc_auc - 1
    brier = float(brier_score_loss(y_true, y_prob))

    lift_top_decile, recall_top_decile = lift_and_recall_at_top_k_pct(y_true, y_prob, 0.10)
    _, recall_top_2_deciles = lift_and_recall_at_top_k_pct(y_true, y_prob, 0.20)

    return ClassificationMetrics(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        ks_statistic=ks,
        gini=gini,
        brier_score=brier,
        precision_at_50=float(precision_score(y_true, y_pred, zero_division=0)),
        recall_at_50=float(recall_score(y_true, y_pred, zero_division=0)),
        f1_at_50=float(f1_score(y_true, y_pred, zero_division=0)),
        lift_at_top_decile=lift_top_decile,
        recall_at_top_decile=recall_top_decile,
        recall_at_top_2_deciles=recall_top_2_deciles,
        n_samples=int(len(y_true)),
        default_rate=float(y_true.mean()),
    )


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict[str, list[float]]:
    """Bucketed observed-vs-predicted default rate, for calibration plots."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    mean_predicted = []
    mean_observed = []
    counts = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        mean_predicted.append(float(y_prob[mask].mean()))
        mean_observed.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))
    return {"mean_predicted": mean_predicted, "mean_observed": mean_observed, "counts": counts}
