"""
tests/test_signals.py
=====================
Tests for the signal-driven allocation bridge. The headline test
(`test_walk_forward_signal_has_no_lookahead`) proves the most important
property: a signal value on date *t* does not depend on any data after *t*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio_optimization import rolling_rebalance_backtest
from src.signals import (
    build_signal_panel,
    signal_mu_provider,
    signal_tilt_transform,
    walk_forward_predictions,
)

TICKERS = ["AAPL", "MSFT", "GOOGL"]


# ---------------------------------------------------------------------------
# Walk-forward generation
# ---------------------------------------------------------------------------
def test_walk_forward_predictions_are_out_of_sample(synthetic_long):
    wf = walk_forward_predictions(
        synthetic_long, "AAPL", engine="linear", refit_freq=21, lookback=252
    )
    assert "pred_logret" in wf
    assert len(wf) > 0
    # Predictions only begin after the first lookback window is available.
    assert not wf["pred_logret"].isna().any()


def test_walk_forward_signal_has_no_lookahead(synthetic_long):
    """
    Re-running the generator on data truncated to an earlier cutoff must yield
    the SAME predictions for the overlapping dates. If future bars influenced
    past signals, truncation would change them — so equality proves causality.
    """
    full = walk_forward_predictions(
        synthetic_long, "AAPL", engine="linear", refit_freq=21, lookback=252
    )["pred_logret"]

    # Truncate the raw data to ~75% of the timeline and regenerate.
    all_dates = synthetic_long.index.get_level_values("Date").unique().sort_values()
    cut = all_dates[int(len(all_dates) * 0.75)]
    truncated_long = synthetic_long[
        synthetic_long.index.get_level_values("Date") <= cut
    ]
    trunc = walk_forward_predictions(
        truncated_long, "AAPL", engine="linear", refit_freq=21, lookback=252
    )["pred_logret"]

    # On the overlap (dates present in both, before the cut) they must match.
    overlap = full.index.intersection(trunc.index)
    overlap = overlap[overlap <= cut]
    assert len(overlap) > 20  # a meaningful overlap exists
    pd.testing.assert_series_equal(
        full.loc[overlap], trunc.loc[overlap], check_names=False, rtol=1e-6
    )


def test_build_signal_panel_shapes(synthetic_long):
    pred, prob = build_signal_panel(
        synthetic_long, TICKERS, engine="linear", refit_freq=21, lookback=252
    )
    assert list(pred.columns) == TICKERS
    assert len(pred) > 0
    if not prob.empty:
        assert list(prob.columns) == TICKERS


# ---------------------------------------------------------------------------
# Optimiser hooks
# ---------------------------------------------------------------------------
def test_signal_mu_provider_returns_annualised_vector(synthetic_long):
    pred, _ = build_signal_panel(synthetic_long, TICKERS, engine="linear",
                                 refit_freq=21, lookback=252)
    provider = signal_mu_provider(pred, window=21)
    # Before any signal exists -> None (backtest falls back to history).
    assert provider(pred.index.min() - pd.Timedelta(days=10), None) is None
    # After signals exist -> a μ vector over the tickers.
    mu = provider(pred.index[-1], None)
    assert mu is not None
    assert set(mu.index) == set(TICKERS)
    # Annualised mean ≈ trailing daily mean × 252.
    expected = pred.tail(21).mean() * 252
    pd.testing.assert_series_equal(mu, expected, check_names=False)


def test_signal_tilt_keeps_weights_on_the_simplex(synthetic_long):
    pred, _ = build_signal_panel(synthetic_long, TICKERS, engine="linear",
                                 refit_freq=21, lookback=252)
    tilt = signal_tilt_transform(pred, tilt_strength=0.5, window=5)
    base = pd.Series(1.0 / len(TICKERS), index=TICKERS)
    tilted = tilt(pred.index[-1], base, None)
    assert tilted.sum() == pytest.approx(1.0, abs=1e-9)
    assert (tilted >= -1e-9).all()           # stays long-only


# ---------------------------------------------------------------------------
# Integration with the backtest engine
# ---------------------------------------------------------------------------
def test_mu_provider_changes_allocation_vs_historical(synthetic_long):
    """A signal-driven backtest should run and (generally) differ from the
    pure-historical max-Sharpe allocation."""
    pred, _ = build_signal_panel(synthetic_long, TICKERS, engine="linear",
                                 refit_freq=21, lookback=126)
    provider = signal_mu_provider(pred, window=21)

    bt_signal = rolling_rebalance_backtest(
        synthetic_long, tickers=TICKERS, strategy="max_sharpe",
        frequency="M", lookback_days=126, cost_rate=0.0, mu_provider=provider,
    )
    bt_hist = rolling_rebalance_backtest(
        synthetic_long, tickers=TICKERS, strategy="max_sharpe",
        frequency="M", lookback_days=126, cost_rate=0.0,
    )
    assert bt_signal.n_rebalances > 0
    assert np.allclose(bt_signal.weights_history.sum(axis=1), 1.0, atol=1e-4)
    # The two weight histories should not be identical everywhere (the signal
    # genuinely moved μ for at least some rebalances).
    common = bt_signal.weights_history.index.intersection(bt_hist.weights_history.index)
    assert not bt_signal.weights_history.loc[common].equals(
        bt_hist.weights_history.loc[common]
    )
