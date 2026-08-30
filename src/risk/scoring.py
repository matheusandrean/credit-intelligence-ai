"""Probability-of-Default to illustrative credit score conversion.

Implements the standard "points-to-double-the-odds" (PDO) scorecard scaling
used across the credit industry to turn a PD into a human-readable score.
The resulting 300-900 score and A-E risk bands are DEMONSTRATIVE ONLY: the
band cut-offs are not a real commercial policy (see configs/project_config.yaml
and RESPONSIBLE_AI.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.config import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "configs" / "project_config.yaml"

PD_FLOOR = 1e-4
PD_CEILING = 1 - 1e-4


@dataclass(frozen=True)
class ScoreCardConfig:
    min_score: int
    max_score: int
    reference_pd: float
    reference_score: int
    pdo: int


@dataclass(frozen=True)
class RiskBandDefinition:
    band: str
    min_score: int
    max_score: int
    description: str


def load_scorecard_config(config_path: Path = CONFIG_PATH) -> ScoreCardConfig:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    score_cfg = raw["score"]
    return ScoreCardConfig(
        min_score=score_cfg["min_score"],
        max_score=score_cfg["max_score"],
        reference_pd=score_cfg["reference_pd"],
        reference_score=score_cfg["reference_score"],
        pdo=score_cfg["pdo"],
    )


def load_risk_bands(config_path: Path = CONFIG_PATH) -> list[RiskBandDefinition]:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return [
        RiskBandDefinition(
            band=b["band"],
            min_score=b["min_score"],
            max_score=b["max_score"],
            description=b["description"],
        )
        for b in raw["risk_bands"]
    ]


def pd_to_score(
    pd_values: np.ndarray | pd.Series | float, cfg: ScoreCardConfig | None = None
) -> np.ndarray:
    """Convert Probability of Default into an illustrative 300-900 score.

    Uses the classic scorecard formula: score = offset + factor * ln(odds),
    where odds is the good:bad odds (1-PD)/PD, `factor = PDO/ln(2)` and
    `offset` is anchored so `reference_pd` maps to `reference_score`.
    Higher score = lower risk (industry convention).
    """
    cfg = cfg or load_scorecard_config()
    pd_arr = np.clip(np.asarray(pd_values, dtype=float), PD_FLOOR, PD_CEILING)

    factor = cfg.pdo / math.log(2)
    reference_odds = (1 - cfg.reference_pd) / cfg.reference_pd
    offset = cfg.reference_score - factor * math.log(reference_odds)

    odds = (1 - pd_arr) / pd_arr
    score = offset + factor * np.log(odds)
    return np.clip(score, cfg.min_score, cfg.max_score).round(0)


def score_to_band(
    scores: np.ndarray | pd.Series, bands: list[RiskBandDefinition] | None = None
) -> np.ndarray:
    """Map illustrative scores to demonstrative A-E risk bands."""
    bands = bands or load_risk_bands()
    scores_arr = np.asarray(scores, dtype=float)
    result = np.full(scores_arr.shape, "UNSCORED", dtype=object)
    for b in bands:
        mask = (scores_arr >= b.min_score) & (scores_arr <= b.max_score)
        result[mask] = b.band
    return result


def pd_to_band(pd_values: np.ndarray | pd.Series | float) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: PD -> (score, band) in one call."""
    scores = pd_to_score(pd_values)
    bands = score_to_band(scores)
    return scores, bands
