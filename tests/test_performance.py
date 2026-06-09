"""
tests/test_performance.py
=========================
Closed-form checks of the risk-adjusted performance metrics on hand-constructed
return series with known answers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.performance import (
    annualised_return,
    annualised_volatility,
    calmar_ratio,
    equity_curve,
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

TD = config.TRADING_DAYS_PER_YEAR


def test_equity_curve_compounds():
    r = pd.Series([0.10, 0.10])
    curve = equity_curve(r)
    assert curve.iloc[0] == pytest.approx(1.10)
    assert curve.iloc[1] == pytest.approx(1.21)


def test_annualised_return_geometric():
    r = pd.Series([0.001] * 252)  # one year of a constant daily return
    expected = (1.001) ** 252 - 1.0
    assert annualised_return(r) == pytest.approx(expected, rel=1e-9)


def test_annualised_volatility_scales_by_sqrt_time():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 1000))
    assert annualised_volatility(r) == pytest.approx(
        r.std(ddof=0) * np.sqrt(TD), rel=1e-9
    )


def test_sharpe_exact_with_zero_risk_free():
    # mean = 0.02, population std = 0.01  ->  sharpe = 2 * sqrt(252).
    r = pd.Series([0.01, 0.03])
    expected = 0.02 / 0.01 * np.sqrt(TD)
    assert sharpe_ratio(r, risk_free=0.0) == pytest.approx(expected, rel=1e-9)


def test_sharpe_is_zero_when_no_variance():
    r = pd.Series([0.005] * 50)  # constant => std 0 => guarded to 0.0
    assert sharpe_ratio(r, risk_free=0.0) == 0.0


def test_sortino_only_penalises_downside():
    # excess = [0.02, -0.01]; downside_dev = sqrt(mean([0.01^2])) = 0.01.
    r = pd.Series([0.02, -0.01])
    expected = r.mean() / 0.01 * np.sqrt(TD)
    assert sortino_ratio(r, risk_free=0.0) == pytest.approx(expected, rel=1e-9)


def test_sortino_is_zero_without_downside():
    r = pd.Series([0.01, 0.02, 0.03])  # never negative vs rf=0
    assert sortino_ratio(r, risk_free=0.0) == 0.0


def test_maximum_drawdown_exact():
    # 1 -> 1.1 -> 0.55 : trough is 50% below the 1.1 peak.
    r = pd.Series([0.10, -0.50])
    assert maximum_drawdown(r) == pytest.approx(-0.50)


def test_maximum_drawdown_is_zero_for_monotonic_growth():
    r = pd.Series([0.01, 0.02, 0.01, 0.03])  # never declines
    assert maximum_drawdown(r) == pytest.approx(0.0)


def test_calmar_equals_return_over_abs_drawdown():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0005, 0.01, 400))
    mdd = maximum_drawdown(r)
    if mdd != 0:
        assert calmar_ratio(r) == pytest.approx(annualised_return(r) / abs(mdd))
