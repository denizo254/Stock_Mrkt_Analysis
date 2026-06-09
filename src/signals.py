"""
src/signals.py  —  BRIDGE: Phase 4 (Modeling) → Phase 5 (Allocation)
====================================================================
Signal-driven allocation: turn the predictive models into a time-varying
expected-return signal that drives the portfolio optimiser, instead of relying
solely on historical mean returns.

The cardinal requirement is the same as everywhere else in this project —
**no look-ahead**. A signal used to set weights on date *t* must come from a
model that was trained only on data observable before *t*. We therefore
generate predictions with a **walk-forward refit**:

    every `refit_freq` days:
        train a model on the trailing `lookback` window  (data strictly < cut)
        predict the next `refit_freq` days                (out-of-sample block)

Concatenating those blocks yields a fully out-of-sample prediction series.
Unlike the Phase-4 models (tuned once via WFV grid search), the signal models
are refit cheaply with fixed hyper-parameters on each window — appropriate for
a rolling backtest where re-searching the grid thousands of times is wasteful.

Two ways to consume the signal (both provided):
  * **μ-driven**   — feed the (annualised) predicted returns as the optimiser's
    μ vector and solve max-Sharpe. (`signal_mu_provider`)
  * **tilt**       — start from a base allocation and overweight high-signal
    names, cross-sectionally. (`signal_tilt_transform`)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.feature_engineering import build_features_for_ticker, feature_columns
from src.modeling import _HAS_XGB, _build_classifier, _build_regressor
from src.portfolio_optimization import rolling_rebalance_backtest
from src.utils import banner, get_logger, timed

logger = get_logger("signals")

TD = config.TRADING_DAYS_PER_YEAR


# ===========================================================================
# Walk-forward out-of-sample prediction generation
# ===========================================================================
def walk_forward_predictions(
    long: pd.DataFrame,
    ticker: str,
    engine: str = "linear",
    refit_freq: int = 21,
    lookback: int = 252,
    include_proba: bool = True,
) -> pd.DataFrame:
    """
    Generate out-of-sample next-day predictions for one ticker by periodic
    walk-forward refitting.

    Returns a DataFrame indexed by date with columns:
      * ``pred_logret`` — predicted next-day log return (regression signal)
      * ``prob_up``     — predicted P(next day up) (classification signal),
                          present only if ``include_proba`` and both classes
                          appear in the training window.

    Every prediction for the block ``[i, i+refit_freq)`` is produced by a model
    trained on ``[i-lookback, i)`` — so the value on any date depends solely on
    earlier data. (This invariant is asserted in the test suite.)
    """
    feats = build_features_for_ticker(long, ticker)
    cols = feature_columns(feats)
    X, y_reg, y_clf = feats[cols], feats["target_logret"], feats["target_dir"]
    n = len(feats)

    pred_logret = pd.Series(np.nan, index=feats.index, dtype=float)
    prob_up = pd.Series(np.nan, index=feats.index, dtype=float)

    if n <= lookback:
        logger.warning("[%s] not enough rows (%d) for lookback=%d; no signal.",
                       ticker, n, lookback)
        return pd.DataFrame({"pred_logret": pred_logret})

    with timed(f"[{ticker}] walk-forward signal ({engine}, refit={refit_freq}d)", logger):
        i = lookback
        while i < n:
            tr = slice(i - lookback, i)          # trailing window, strictly < i
            blk = slice(i, min(i + refit_freq, n))  # out-of-sample block
            X_tr, X_blk = X.iloc[tr], X.iloc[blk]

            reg, _ = _build_regressor(engine)
            reg.fit(X_tr, y_reg.iloc[tr])
            pred_logret.iloc[blk] = reg.predict(X_blk)

            if include_proba and y_clf.iloc[tr].nunique() == 2:
                clf, _ = _build_classifier(engine)
                clf.fit(X_tr, y_clf.iloc[tr])
                prob_up.iloc[blk] = clf.predict_proba(X_blk)[:, 1]

            i += refit_freq

    out = pd.DataFrame({"pred_logret": pred_logret})
    if include_proba:
        out["prob_up"] = prob_up
    return out.dropna(subset=["pred_logret"])


def build_signal_panel(
    long: pd.DataFrame,
    tickers: list[str] | None = None,
    engine: str = "linear",
    refit_freq: int = 21,
    lookback: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build cross-sectional out-of-sample signal panels for the universe.

    Returns ``(pred_logret_panel, prob_up_panel)`` — each a DataFrame indexed
    by date with one column per ticker.
    """
    tickers = tickers or config.TICKERS
    logret_cols: dict[str, pd.Series] = {}
    prob_cols: dict[str, pd.Series] = {}
    for t in tickers:
        wf = walk_forward_predictions(
            long, t, engine=engine, refit_freq=refit_freq, lookback=lookback
        )
        logret_cols[t] = wf["pred_logret"]
        if "prob_up" in wf:
            prob_cols[t] = wf["prob_up"]

    pred_panel = pd.DataFrame(logret_cols).sort_index()
    prob_panel = pd.DataFrame(prob_cols).sort_index() if prob_cols else pd.DataFrame()
    logger.info(
        "Signal panel built: %d dates × %d tickers (%s → %s)",
        len(pred_panel), pred_panel.shape[1],
        pred_panel.index.min().date() if len(pred_panel) else "—",
        pred_panel.index.max().date() if len(pred_panel) else "—",
    )
    return pred_panel, prob_panel


# ===========================================================================
# Optimiser hooks: μ-provider and weight-tilt
# ===========================================================================
def signal_mu_provider(pred_logret_panel: pd.DataFrame, window: int = 21):
    """
    Build a ``mu_provider`` callable for ``rolling_rebalance_backtest``.

    At each rebalance date it averages the trailing ``window`` of out-of-sample
    predicted log returns (data up to that date only) and annualises them into
    an expected-return vector μ. Returns ``None`` before signals are available,
    so the backtest cleanly falls back to the historical mean.
    """

    def provider(date, _window_log_returns) -> pd.Series | None:
        sub = pred_logret_panel.loc[:date]
        if sub.empty:
            return None
        recent = sub.tail(window).mean()
        if recent.isna().all():
            return None
        return recent * TD  # annualise daily expected log return

    return provider


def signal_tilt_transform(
    pred_logret_panel: pd.DataFrame,
    tilt_strength: float = 0.5,
    window: int = 5,
):
    """
    Build a ``weight_transform`` callable that tilts a base allocation toward
    names with a strong recent signal.

        w_i  ->  w_i · (1 + k · z_i),   renormalised, clipped to ≥ 0

    where ``z_i`` is the cross-sectional z-score of the trailing-mean predicted
    return across assets as of ``date``. ``tilt_strength`` (k) controls how
    aggressively conviction overrides the base weights. Uses only data ≤ date.
    """

    def transform(date, base_weights: pd.Series, _window_log_returns) -> pd.Series:
        sub = pred_logret_panel.loc[:date].tail(window)
        if sub.empty:
            return base_weights
        signal = sub.mean().reindex(base_weights.index)
        if signal.isna().all():
            return base_weights
        signal = signal.fillna(0.0)
        std = signal.std()
        z = (signal - signal.mean()) / (std + 1e-9)
        tilted = base_weights * (1.0 + tilt_strength * z)
        tilted = tilted.clip(lower=0.0)                  # stay long-only
        total = tilted.sum()
        if total <= 0:
            return base_weights
        return tilted / total

    return transform


# ===========================================================================
# High-level runner
# ===========================================================================
def run_signal_strategies(
    long: pd.DataFrame,
    tickers: list[str] | None = None,
    frequency: str = config.REBALANCE.frequency,
    lookback_days: int = config.REBALANCE.lookback_days,
    cost_rate: float = config.REBALANCE.cost_rate,
    engine: str = "linear",
    refit_freq: int = 21,
    signal_window: int = 21,
    tilt_strength: float = 0.5,
) -> tuple[dict, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Build the out-of-sample signal panel once and run the two signal-driven
    strategies through the rolling rebalancing engine.

    Returns ``(backtests, (pred_panel, prob_panel))`` where ``backtests`` maps
    a label to a ``RollingBacktestResult``, ready to drop into
    ``performance.dynamic_benchmark_report`` alongside the pure-MPT strategies.
    """
    tickers = tickers or config.TICKERS
    logger.info(banner("SIGNAL-DRIVEN ALLOCATION — walk-forward signal panel"))

    pred_panel, prob_panel = build_signal_panel(
        long, tickers, engine=engine, refit_freq=refit_freq, lookback=lookback_days
    )

    mu_provider = signal_mu_provider(pred_panel, window=signal_window)
    tilt = signal_tilt_transform(pred_panel, tilt_strength=tilt_strength, window=signal_window)

    bt_mu = rolling_rebalance_backtest(
        long, tickers=tickers, strategy="max_sharpe", frequency=frequency,
        lookback_days=lookback_days, cost_rate=cost_rate, mu_provider=mu_provider,
    )
    bt_tilt = rolling_rebalance_backtest(
        long, tickers=tickers, strategy="equal_weight", frequency=frequency,
        lookback_days=lookback_days, cost_rate=cost_rate, weight_transform=tilt,
    )
    return (
        {"Signal μ (Max Sharpe)": bt_mu, "Signal Tilt (EW base)": bt_tilt},
        (pred_panel, prob_panel),
    )
