"""
tests/conftest.py
=================
Shared pytest fixtures.

The whole suite runs on **synthetic, deterministic** data — no network, no
yfinance calls — so it is fast and reproducible in CI. The synthetic frames
mirror the exact schema the pipeline expects from
``data_ingestion.download_market_data`` (a tidy long frame indexed by
(Date, Ticker) with OHLCV columns), so every downstream module is exercised
against a realistic shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_ingestion import OHLCV_COLUMNS

SYNTH_TICKERS = ["AAPL", "MSFT", "GOOGL", "SPY"]
N_DAYS = 800  # enough to clear the 200-day SMA + 63-day beta warm-up + target shift


def _make_one(symbol: str, seed: int) -> pd.DataFrame:
    """Generate a single ticker's OHLCV frame from a seeded geometric walk."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=N_DAYS)
    drift = rng.uniform(0.0002, 0.0006)
    rets = rng.normal(drift, 0.015, N_DAYS)
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1.0 + rng.uniform(0.0, 0.012, N_DAYS))
    low = close * (1.0 - rng.uniform(0.0, 0.012, N_DAYS))
    open_ = close * (1.0 + rng.normal(0.0, 0.005, N_DAYS))
    volume = rng.integers(1_000_000, 5_000_000, N_DAYS).astype(float)
    frame = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    frame["Date"] = dates
    frame["Ticker"] = symbol
    return frame


@pytest.fixture(scope="session")
def synthetic_long() -> pd.DataFrame:
    """Tidy long OHLCV frame for several tickers + SPY (session-cached)."""
    frames = [_make_one(sym, seed=i + 1) for i, sym in enumerate(SYNTH_TICKERS)]
    long = pd.concat(frames).set_index(["Date", "Ticker"]).sort_index()
    return long[OHLCV_COLUMNS]


@pytest.fixture(scope="session")
def feature_matrix(synthetic_long):
    """A built, dense feature+target matrix for a single ticker."""
    from src.feature_engineering import build_features_for_ticker

    return build_features_for_ticker(synthetic_long, "AAPL")


@pytest.fixture()
def mu_cov():
    """A small, fixed (μ, Σ) pair for deterministic portfolio-math assertions."""
    tickers = ["A", "B", "C"]
    mu = pd.Series([0.10, 0.18, 0.14], index=tickers)
    cov = pd.DataFrame(
        [
            [0.0400, 0.0050, 0.0010],
            [0.0050, 0.0900, 0.0020],
            [0.0010, 0.0020, 0.0625],
        ],
        index=tickers,
        columns=tickers,
    )
    return mu, cov
