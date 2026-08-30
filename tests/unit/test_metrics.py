"""Unit tests for src.models.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.metrics import (
    compute_classification_metrics,
    ks_statistic,
    lift_and_recall_at_top_k_pct,
    reliability_curve,
)


def _synthetic_scores(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_true = rng.binomial(1, 0.1, size=n)
    # Scores correlated with the label but noisy - a "decent but imperfect" model.
    y_prob = np.clip(0.05 + 0.6 * y_true + rng.normal(0, 0.15, size=n), 0, 1)
    return y_true, y_prob


def test_perfect_classifier_has_auc_one_and_ks_one() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.01, 0.02, 0.03, 0.9, 0.95, 0.99])
    metrics = compute_classification_metrics(y_true, y_prob)
    assert metrics.roc_auc == 1.0
    assert metrics.gini == 1.0
    assert ks_statistic(y_true, y_prob) == 1.0


def test_random_classifier_has_auc_near_half() -> None:
    rng = np.random.default_rng(1)
    y_true = rng.binomial(1, 0.5, size=5000)
    y_prob = rng.random(5000)  # unrelated to y_true
    metrics = compute_classification_metrics(y_true, y_prob)
    assert 0.45 <= metrics.roc_auc <= 0.55


def test_decent_classifier_metrics_are_internally_consistent() -> None:
    y_true, y_prob = _synthetic_scores()
    metrics = compute_classification_metrics(y_true, y_prob)
    assert 0.5 < metrics.roc_auc < 1.0
    assert metrics.gini == pytest.approx(2 * metrics.roc_auc - 1)
    assert 0.0 <= metrics.brier_score <= 1.0
    assert metrics.n_samples == len(y_true)
    assert metrics.lift_at_top_decile >= 1.0  # a working model beats random targeting


def test_lift_and_recall_at_top_k_pct_matches_manual_computation() -> None:
    y_true = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.1, 0.2, 0.1, 0.8, 0.3, 0.1, 0.2, 0.1, 0.1])
    lift, recall = lift_and_recall_at_top_k_pct(y_true, y_prob, pct=0.2)
    # top 20% = top 2 scores -> indices 0 and 4, both true positives
    assert recall == 1.0
    assert lift == pytest.approx(1 / 0.2)


def test_reliability_curve_shapes_match() -> None:
    y_true, y_prob = _synthetic_scores()
    curve = reliability_curve(y_true, y_prob, n_bins=10)
    assert len(curve["mean_predicted"]) == len(curve["mean_observed"]) == len(curve["counts"])
    assert sum(curve["counts"]) == len(y_true)
