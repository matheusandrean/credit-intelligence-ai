"""Unit tests for src.risk.scoring."""

from __future__ import annotations

import numpy as np

from src.risk.scoring import (
    RiskBandDefinition,
    ScoreCardConfig,
    load_risk_bands,
    pd_to_band,
    pd_to_score,
    score_to_band,
)

DEFAULT_CFG = ScoreCardConfig(
    min_score=300, max_score=900, reference_pd=0.08, reference_score=660, pdo=40
)


def test_reference_pd_maps_to_reference_score() -> None:
    score = pd_to_score(DEFAULT_CFG.reference_pd, DEFAULT_CFG)
    assert abs(float(np.atleast_1d(score)[0]) - DEFAULT_CFG.reference_score) < 1.0


def test_lower_pd_yields_higher_score() -> None:
    low_risk_score = pd_to_score(0.01, DEFAULT_CFG)
    high_risk_score = pd_to_score(0.30, DEFAULT_CFG)
    assert float(np.atleast_1d(low_risk_score)[0]) > float(np.atleast_1d(high_risk_score)[0])


def test_scores_are_bounded() -> None:
    extreme_pds = np.array([1e-6, 0.999999])
    scores = pd_to_score(extreme_pds, DEFAULT_CFG)
    assert scores.min() >= DEFAULT_CFG.min_score
    assert scores.max() <= DEFAULT_CFG.max_score


def test_score_to_band_covers_full_range() -> None:
    bands = load_risk_bands()
    scores = np.array([300, 550, 650, 730, 800, 900])
    band_labels = score_to_band(scores, bands)
    assert "UNSCORED" not in band_labels


def test_score_to_band_is_monotonic_with_risk() -> None:
    bands = [
        RiskBandDefinition("E", 300, 539, ""),
        RiskBandDefinition("D", 540, 619, ""),
        RiskBandDefinition("C", 620, 699, ""),
        RiskBandDefinition("B", 700, 779, ""),
        RiskBandDefinition("A", 780, 900, ""),
    ]
    assert score_to_band(np.array([320]), bands)[0] == "E"
    assert score_to_band(np.array([850]), bands)[0] == "A"


def test_pd_to_band_end_to_end() -> None:
    pds = np.array([0.005, 0.05, 0.15, 0.40])
    scores, bands = pd_to_band(pds)
    assert len(scores) == len(bands) == 4
    # riskiest PD should map to the worst (last, highest-risk) band ordering
    assert bands[-1] in {"D", "E"}
    assert bands[0] in {"A", "B"}
