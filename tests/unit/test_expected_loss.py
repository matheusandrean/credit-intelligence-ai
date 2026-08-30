"""Unit tests for src.risk.expected_loss."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk.expected_loss import (
    compute_ead,
    compute_expected_loss,
    compute_lgd,
    portfolio_expected_loss_summary,
)


def test_lgd_bounded_between_0_and_1() -> None:
    lgd = compute_lgd(5000)
    assert (lgd >= 0).all()
    assert (lgd <= 1).all()


def test_ead_uses_drawn_plus_ccf_of_undrawn() -> None:
    limit = np.array([1000.0])
    balance = np.array([400.0])
    ead = compute_ead(limit, balance)
    # 400 drawn + 0.5 * (1000-400) undrawn = 700
    assert ead[0] == 700.0


def test_ead_handles_missing_balance() -> None:
    limit = np.array([1000.0])
    balance = np.array([np.nan])
    ead = compute_ead(limit, balance)
    assert ead[0] == 500.0  # treated as 0 balance -> 0.5 * 1000


def test_expected_loss_is_pd_times_lgd_times_ead() -> None:
    pd_values = pd.Series([0.1, 0.2])
    limit = pd.Series([1000.0, 2000.0])
    balance = pd.Series([500.0, 1000.0])
    el_df = compute_expected_loss(pd_values, limit, balance, seed=1)
    assert np.allclose(el_df["expected_loss"], el_df["pd"] * el_df["lgd"] * el_df["ead"])


def test_portfolio_summary_aggregates_correctly() -> None:
    el_df = pd.DataFrame(
        {
            "pd": [0.1, 0.2],
            "lgd": [0.6, 0.6],
            "ead": [1000.0, 2000.0],
            "expected_loss": [60.0, 240.0],
        }
    )
    summary = portfolio_expected_loss_summary(el_df)
    assert summary["total_exposure"] == 3000.0
    assert summary["total_expected_loss"] == 300.0
    assert summary["expected_loss_rate"] == 0.1
    assert summary["n_accounts"] == 2
