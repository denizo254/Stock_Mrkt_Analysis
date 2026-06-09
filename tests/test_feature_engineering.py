"""
tests/test_feature_engineering.py
=================================
Verifies the indicator math and — critically — that lagged features and the
prediction target are aligned so that no future information leaks into a
feature row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_lagged_features,
    average_true_range,
    bollinger_bands,
    build_features_for_ticker,
    feature_columns,
    log_returns,
    rsi,
)


# ---------------------------------------------------------------------------
# Pure indicator math
# ---------------------------------------------------------------------------
def test_log_returns_exact():
    close = pd.Series([100.0, 110.0, 99.0])
    lr = log_returns(close)
    assert np.isnan(lr.iloc[0])
    assert lr.iloc[1] == pytest.approx(np.log(110 / 100))
    assert lr.iloc[2] == pytest.approx(np.log(99 / 110))


def test_lagged_features_pull_from_the_past_only():
    df = pd.DataFrame({"x": np.arange(10, dtype=float)})
    out = add_lagged_features(df.copy(), ["x"], (1, 2, 5))
    # A lag-k feature at row i must equal the value k rows earlier.
    assert out["x_lag1"].iloc[3] == out["x"].iloc[2]
    assert out["x_lag2"].iloc[7] == out["x"].iloc[5]
    assert out["x_lag5"].iloc[5] == out["x"].iloc[0]
    # The first k rows are NaN (no past available) — never forward-filled.
    assert out["x_lag5"].iloc[:5].isna().all()


def test_rsi_is_bounded_0_100():
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    r = rsi(close, period=14).dropna()
    assert len(r) > 0
    assert (r >= 0).all() and (r <= 100).all()


def test_rsi_saturates_to_100_on_pure_uptrend():
    close = pd.Series(np.linspace(100, 200, 60))  # monotonic up => no losses
    r = rsi(close, period=14).dropna()
    assert r.iloc[-1] == pytest.approx(100.0)


def test_bollinger_percent_b_matches_definition():
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 100))))
    bb = bollinger_bands(close, window=20, k=2.0)
    # %b = (close - lower) / (upper - lower), recomputed independently.
    expected = (close - bb["bb_lower"]) / (bb["bb_upper"] - bb["bb_lower"])
    pd.testing.assert_series_equal(
        bb["bb_pct_b"].dropna(), expected.dropna(), check_names=False
    )


def test_atr_is_nonnegative():
    rng = np.random.default_rng(2)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120))))
    high = close * 1.01
    low = close * 0.99
    atr = average_true_range(high, low, close, period=14).dropna()
    assert (atr >= 0).all()


# ---------------------------------------------------------------------------
# Assembled matrix: target alignment & leakage guards
# ---------------------------------------------------------------------------
def test_feature_matrix_has_no_nans(feature_matrix):
    assert not feature_matrix.isna().any().any()


def test_target_direction_is_sign_of_next_day_return(feature_matrix):
    expected = (feature_matrix["target_logret"] > 0).astype("int8")
    pd.testing.assert_series_equal(
        feature_matrix["target_dir"], expected, check_names=False
    )


def test_feature_columns_exclude_targets(feature_matrix):
    cols = feature_columns(feature_matrix)
    assert "target_logret" not in cols
    assert "target_dir" not in cols
    assert len(cols) > 10  # a rich feature set was actually built


def test_target_is_strictly_forward_looking(synthetic_long):
    """
    target_logret at date t must equal the log return realised from t to t+1,
    i.e. tomorrow's return — the only forward-looking column in the matrix.
    """
    from src.data_ingestion import get_ticker_frame

    feats = build_features_for_ticker(synthetic_long, "AAPL")
    close = get_ticker_frame(synthetic_long, "AAPL")["Adj Close"]
    realised_next = np.log(close / close.shift(1)).shift(-1)
    # Compare on the dense matrix's index.
    aligned = realised_next.reindex(feats.index)
    pd.testing.assert_series_equal(
        feats["target_logret"], aligned, check_names=False, rtol=1e-9
    )
