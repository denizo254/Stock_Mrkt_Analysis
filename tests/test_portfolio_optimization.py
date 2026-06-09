"""
tests/test_portfolio_optimization.py
====================================
Verifies the MPT algebra, the optimiser invariants (weights sum to 1, bounds
respected, min-variance really is minimal), the efficient-frontier geometry,
and the rolling backtest's structural cost/turnover properties.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio_optimization import (
    annualised_cov,
    annualised_mean,
    compute_returns,
    efficient_frontier,
    maximum_sharpe_portfolio,
    minimum_variance_portfolio,
    portfolio_performance,
    rolling_rebalance_backtest,
)

TOL = 1e-6


# ---------------------------------------------------------------------------
# Core algebra
# ---------------------------------------------------------------------------
def test_portfolio_performance_matches_closed_form():
    mu = np.array([0.10, 0.20])
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    w = np.array([0.5, 0.5])
    ret, vol, sharpe = portfolio_performance(w, mu, cov, risk_free=0.0)
    assert ret == pytest.approx(0.15)
    assert vol == pytest.approx(np.sqrt(0.25 * 0.04 + 0.25 * 0.09))
    assert sharpe == pytest.approx(0.15 / vol)


# ---------------------------------------------------------------------------
# Optimiser invariants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("optimizer", [maximum_sharpe_portfolio, minimum_variance_portfolio])
def test_weights_are_long_only_and_sum_to_one(mu_cov, optimizer):
    mu, cov = mu_cov
    stats = optimizer(mu, cov, risk_free=0.0)
    assert stats.weights.sum() == pytest.approx(1.0, abs=1e-5)
    assert (stats.weights >= -TOL).all()           # long-only lower bound
    assert (stats.weights <= 1.0 + TOL).all()       # upper bound
    assert not stats.weights.isna().any()


def test_min_variance_has_lowest_variance(mu_cov):
    mu, cov = mu_cov
    mv = minimum_variance_portfolio(mu, cov, risk_free=0.0)
    ms = maximum_sharpe_portfolio(mu, cov, risk_free=0.0)
    # By construction the GMV portfolio's variance cannot exceed any other
    # feasible portfolio's, including the max-Sharpe portfolio.
    assert mv.volatility <= ms.volatility + 1e-4
    # And it should not be riskier than the least-risky single asset.
    asset_vols = np.sqrt(np.diag(cov.to_numpy()))
    assert mv.volatility <= asset_vols.min() + 1e-4


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------
def test_efficient_frontier_is_left_bounded_by_min_variance(mu_cov):
    mu, cov = mu_cov
    mv = minimum_variance_portfolio(mu, cov, risk_free=0.0)
    frontier = efficient_frontier(mu, cov, n_points=25, risk_free=0.0)
    assert len(frontier) == 25
    # No frontier point can be less volatile than the global minimum-variance
    # portfolio (it is, by definition, the leftmost point).
    assert (frontier["volatility"] >= mv.volatility - 1e-4).all()
    # Frontier weights are valid simplex points too.
    weight_cols = [c for c in frontier.columns if c.startswith("w_")]
    assert np.allclose(frontier[weight_cols].sum(axis=1), 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Estimation helpers
# ---------------------------------------------------------------------------
def test_annualisation_scales_by_trading_days(synthetic_long):
    import config

    tickers = ["AAPL", "MSFT", "GOOGL"]
    rets = compute_returns(synthetic_long, tickers)
    assert annualised_mean(rets).index.tolist() == tickers
    # Annualised mean == daily mean × 252.
    np.testing.assert_allclose(
        annualised_mean(rets).to_numpy(),
        rets.mean().to_numpy() * config.TRADING_DAYS_PER_YEAR,
    )
    # Covariance is square and symmetric.
    cov = annualised_cov(rets)
    assert cov.shape == (3, 3)
    np.testing.assert_allclose(cov.to_numpy(), cov.to_numpy().T)


# ---------------------------------------------------------------------------
# Rolling rebalancing backtest
# ---------------------------------------------------------------------------
def test_rolling_backtest_structural_invariants(synthetic_long):
    bt = rolling_rebalance_backtest(
        synthetic_long,
        tickers=["AAPL", "MSFT", "GOOGL"],
        strategy="max_sharpe",
        frequency="M",
        lookback_days=126,
        cost_rate=0.0007,
    )
    assert bt.n_rebalances > 0
    # Each rebalance's target weights form a valid long-only simplex.
    assert np.allclose(bt.weights_history.sum(axis=1), 1.0, atol=1e-4)
    assert (bt.weights_history.to_numpy() >= -1e-6).all()
    # Turnover is non-negative.
    assert (bt.turnover >= 0).all()
    # Costs can only DRAG: the net curve must end at/below the gross curve.
    assert bt.equity_curve.iloc[-1] <= bt.gross_equity_curve.iloc[-1] + 1e-9
    # Equity curve is positive throughout (no nonsensical wipeout on synthetic).
    assert (bt.equity_curve > 0).all()


def test_rolling_backtest_rejects_missing_ticker(synthetic_long):
    """A requested symbol absent from the data raises a clear, early error
    rather than failing cryptically deep in the simulation (cloud robustness)."""
    with pytest.raises(RuntimeError, match="missing"):
        rolling_rebalance_backtest(
            synthetic_long, tickers=["AAPL", "DOES_NOT_EXIST"], frequency="M"
        )


def test_rolling_backtest_zero_cost_equals_gross(synthetic_long):
    """With a 0 bps cost the net and gross equity curves must coincide."""
    bt = rolling_rebalance_backtest(
        synthetic_long,
        tickers=["AAPL", "MSFT", "GOOGL"],
        strategy="min_variance",
        frequency="Q",
        lookback_days=126,
        cost_rate=0.0,
    )
    np.testing.assert_allclose(
        bt.equity_curve.to_numpy(), bt.gross_equity_curve.to_numpy(), rtol=1e-9
    )
    assert bt.total_cost == pytest.approx(0.0)
