"""
tests/test_risk_overlays.py
===========================
Tests for the robustness layer: Ledoit-Wolf covariance shrinkage, the
position cap, volatility targeting, the drawdown stop, and the crucial
reduction property (overlays OFF == the original baseline behaviour).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio_optimization import (
    annualised_cov,
    apply_position_cap,
    compute_returns,
    rolling_rebalance_backtest,
)
from src.performance import maximum_drawdown

TICKERS = ["AAPL", "MSFT", "GOOGL"]


# ---------------------------------------------------------------------------
# Ledoit-Wolf shrinkage
# ---------------------------------------------------------------------------
def test_ledoit_wolf_cov_is_valid_and_differs_from_sample(synthetic_long):
    rets = compute_returns(synthetic_long, TICKERS)
    sample = annualised_cov(rets, method="sample")
    lw = annualised_cov(rets, method="ledoit_wolf")

    assert lw.shape == sample.shape
    # Symmetric.
    np.testing.assert_allclose(lw.to_numpy(), lw.to_numpy().T, atol=1e-12)
    # Positive semi-definite (all eigenvalues > 0 for a shrunk estimate).
    assert (np.linalg.eigvalsh(lw.to_numpy()) > 0).all()
    # Shrinkage actually changed the matrix.
    assert not np.allclose(lw.to_numpy(), sample.to_numpy())


def test_unknown_cov_method_raises(synthetic_long):
    rets = compute_returns(synthetic_long, TICKERS)
    with pytest.raises(ValueError):
        annualised_cov(rets, method="nope")


# ---------------------------------------------------------------------------
# Position cap (water-filling)
# ---------------------------------------------------------------------------
def test_position_cap_respects_limit_and_stays_invested():
    w = pd.Series({"A": 0.7, "B": 0.2, "C": 0.1})
    capped = apply_position_cap(w, cap=0.4)
    assert capped.max() <= 0.4 + 1e-9
    assert capped.sum() == pytest.approx(1.0, abs=1e-9)
    assert (capped >= -1e-12).all()


def test_position_cap_noop_when_already_under_cap():
    w = pd.Series({"A": 0.34, "B": 0.33, "C": 0.33})
    capped = apply_position_cap(w, cap=0.5)
    pd.testing.assert_series_equal(capped, w)


# ---------------------------------------------------------------------------
# Reduction property: overlays OFF must equal the baseline exactly
# ---------------------------------------------------------------------------
def test_overlays_off_matches_baseline(synthetic_long):
    base = rolling_rebalance_backtest(
        synthetic_long, tickers=TICKERS, strategy="max_sharpe",
        frequency="M", lookback_days=126, cost_rate=0.0007,
    )
    # Explicitly pass the "off" sentinels — must be bit-for-bit identical.
    same = rolling_rebalance_backtest(
        synthetic_long, tickers=TICKERS, strategy="max_sharpe",
        frequency="M", lookback_days=126, cost_rate=0.0007,
        cov_method="sample", max_weight=None, target_vol=None, drawdown_stop=None,
    )
    np.testing.assert_allclose(
        base.equity_curve.to_numpy(), same.equity_curve.to_numpy(), rtol=1e-12
    )
    assert base.n_stops == 0
    # Exposure is fully invested throughout when no overlay is active.
    assert np.allclose(same.exposure.to_numpy(), 1.0)


# ---------------------------------------------------------------------------
# Volatility targeting
# ---------------------------------------------------------------------------
def test_target_vol_scales_exposure_and_reduces_realised_vol(synthetic_long):
    full = rolling_rebalance_backtest(
        synthetic_long, tickers=TICKERS, strategy="max_sharpe",
        frequency="M", lookback_days=126, cost_rate=0.0,
    )
    targeted = rolling_rebalance_backtest(
        synthetic_long, tickers=TICKERS, strategy="max_sharpe",
        frequency="M", lookback_days=126, cost_rate=0.0,
        target_vol=0.08,  # a low target relative to single-stock vol
    )
    # Exposure must never exceed the (default) max leverage of 1.0.
    assert targeted.exposure.max() <= 1.0 + 1e-9
    # At least sometimes it de-levers below 1.
    assert targeted.exposure.min() < 1.0
    # Realised annualised vol should drop versus the fully-invested book.
    full_vol = full.daily_returns.std(ddof=0) * np.sqrt(252)
    tgt_vol = targeted.daily_returns.std(ddof=0) * np.sqrt(252)
    assert tgt_vol < full_vol


# ---------------------------------------------------------------------------
# Drawdown stop
# ---------------------------------------------------------------------------
def test_drawdown_stop_triggers_and_caps_losses(synthetic_long):
    no_stop = rolling_rebalance_backtest(
        synthetic_long, tickers=TICKERS, strategy="max_sharpe",
        frequency="M", lookback_days=126, cost_rate=0.0,
    )
    stopped = rolling_rebalance_backtest(
        synthetic_long, tickers=TICKERS, strategy="max_sharpe",
        frequency="M", lookback_days=126, cost_rate=0.0,
        drawdown_stop=0.10, drawdown_reenter=0.05,
    )
    # If the unconstrained book ever breached -10%, the stop must have fired
    # and the de-risked book's max drawdown must be no worse.
    if maximum_drawdown(no_stop.daily_returns) <= -0.10:
        assert stopped.n_stops > 0
        assert (stopped.exposure == 0.0).any()        # went to cash at least once
        assert maximum_drawdown(stopped.daily_returns) >= maximum_drawdown(
            no_stop.daily_returns
        ) - 1e-9
