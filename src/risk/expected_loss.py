"""Expected Loss (EL = PD x LGD x EAD) for the synthetic credit portfolio.

LGD and EAD are conceptual/synthetic approximations meant to demonstrate the
EL framework end-to-end, not a validated loss model - see MODEL_CARD.md and
RESPONSIBLE_AI.md for the explicit limitations disclaimer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.config import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "configs" / "project_config.yaml"

# Credit conversion factor applied to the undrawn portion of the line, per
# a simplified unsecured-revolving-credit convention (Basel-style EAD
# approximation, not a validated regulatory model).
CREDIT_CONVERSION_FACTOR = 0.5


@dataclass(frozen=True)
class ExpectedLossConfig:
    base_lgd: float
    lgd_noise_std: float


def load_expected_loss_config(config_path: Path = CONFIG_PATH) -> ExpectedLossConfig:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    el_cfg = raw["expected_loss"]
    return ExpectedLossConfig(base_lgd=el_cfg["base_lgd"], lgd_noise_std=el_cfg["lgd_noise_std"])


def compute_lgd(n: int, cfg: ExpectedLossConfig | None = None, seed: int = 42) -> np.ndarray:
    """Synthetic Loss Given Default: a base rate plus bounded noise."""
    cfg = cfg or load_expected_loss_config()
    rng = np.random.default_rng(seed)
    lgd = cfg.base_lgd + rng.normal(0, cfg.lgd_noise_std, size=n)
    return np.clip(lgd, 0.05, 0.99)


def compute_ead(
    credit_limit: pd.Series | np.ndarray, revolving_balance: pd.Series | np.ndarray
) -> np.ndarray:
    """Exposure at Default for a revolving line: drawn balance plus a
    conversion-factor share of the undrawn commitment."""
    limit = np.asarray(credit_limit, dtype=float)
    balance = np.asarray(revolving_balance, dtype=float)
    balance = np.nan_to_num(balance, nan=0.0)
    undrawn = np.clip(limit - balance, 0, None)
    return balance + CREDIT_CONVERSION_FACTOR * undrawn


def compute_expected_loss(
    pd_values: pd.Series | np.ndarray,
    credit_limit: pd.Series | np.ndarray,
    revolving_balance: pd.Series | np.ndarray,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute PD, LGD, EAD and Expected Loss for a set of accounts."""
    pd_arr = np.asarray(pd_values, dtype=float)
    n = len(pd_arr)
    lgd = compute_lgd(n, seed=seed)
    ead = compute_ead(credit_limit, revolving_balance)
    expected_loss = pd_arr * lgd * ead
    return pd.DataFrame(
        {
            "pd": pd_arr,
            "lgd": lgd,
            "ead": ead,
            "expected_loss": expected_loss,
        }
    )


def portfolio_expected_loss_summary(el_df: pd.DataFrame) -> dict:
    total_exposure = float(el_df["ead"].sum())
    total_expected_loss = float(el_df["expected_loss"].sum())
    return {
        "total_exposure": total_exposure,
        "total_expected_loss": total_expected_loss,
        "expected_loss_rate": (total_expected_loss / total_exposure if total_exposure > 0 else 0.0),
        "average_pd": float(el_df["pd"].mean()),
        "average_lgd": float(el_df["lgd"].mean()),
        "n_accounts": int(len(el_df)),
    }
