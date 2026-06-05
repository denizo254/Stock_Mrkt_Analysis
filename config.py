"""
config.py
=========
Central, single-source-of-truth configuration for the entire pipeline.

Every module imports from here rather than hard-coding paths, tickers, or
hyper-parameters. Changing a knob in one place re-wires the whole project,
which keeps experiments reproducible and reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
FIGURE_DIR: Path = OUTPUT_DIR / "figures"
MODEL_DIR: Path = OUTPUT_DIR / "models"
REPORT_DIR: Path = OUTPUT_DIR / "reports"

# Directories that the pipeline is allowed to create on demand.
_MANAGED_DIRS = [
    RAW_DIR,
    PROCESSED_DIR,
    FIGURE_DIR,
    MODEL_DIR,
    REPORT_DIR,
]


def ensure_dirs() -> None:
    """Create every managed output directory if it does not already exist."""
    for d in _MANAGED_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Universe & sampling window
# ---------------------------------------------------------------------------
# The investable universe. SPY is held out separately as the market benchmark
# and is NOT part of the optimised portfolio (it is the thing we measure
# ourselves against).
TICKERS: list[str] = ["AAPL", "MSFT", "GOOGL", "NVDA"]
BENCHMARK: str = "SPY"

# All symbols we actually download (universe + benchmark).
ALL_SYMBOLS: list[str] = TICKERS + [BENCHMARK]

# Lookback window. yfinance accepts explicit dates; we default to ~8 years of
# daily data, which comfortably spans multiple regimes (2018 selloff, the 2020
# COVID crash + recovery, the 2022 rate-hike drawdown, the 2023-24 AI rally).
START_DATE: str = "2017-01-01"
END_DATE: str | None = None  # None => "today" (yfinance treats it as live)

DATA_INTERVAL: str = "1d"  # daily bars


# ---------------------------------------------------------------------------
# Risk / return conventions
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR: int = 252

# Annual risk-free rate used in Sharpe / Sortino and the tangency portfolio.
# 4.0% is a reasonable proxy for the recent short-term Treasury yield; expose
# it here so it is trivial to re-run under different rate assumptions.
RISK_FREE_RATE: float = 0.04


# ---------------------------------------------------------------------------
# Feature-engineering windows (Phase 3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    sma_windows: tuple[int, ...] = (20, 50, 200)
    ema_windows: tuple[int, ...] = (20, 50, 200)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_window: int = 20
    bollinger_k: float = 2.0
    atr_period: int = 14
    volatility_window: int = 21        # ~1 trading month
    skew_window: int = 63              # ~1 trading quarter
    beta_window: int = 63              # rolling beta vs benchmark
    return_lags: tuple[int, ...] = (1, 2, 5)
    volume_lags: tuple[int, ...] = (1, 2, 5)


FEATURES = FeatureConfig()


# ---------------------------------------------------------------------------
# Modeling configuration (Phase 4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    # Fraction of the (chronological) sample reserved for the final hold-out
    # test. The remainder is used for walk-forward CV + validation.
    test_size: float = 0.2
    # Number of expanding-window folds for TimeSeriesSplit during tuning.
    n_splits: int = 5
    # Reproducibility.
    random_state: int = 42
    # Which engine to use. "xgboost" or "linear" (Ridge / LogisticRegression).
    regressor: str = "xgboost"
    classifier: str = "xgboost"


MODEL = ModelConfig()


# ---------------------------------------------------------------------------
# Portfolio optimization (Phase 5)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioConfig:
    # Long-only, fully-invested weights by default. Set allow_short=True to
    # relax the lower bound to a negative number.
    allow_short: bool = False
    min_weight: float = 0.0
    max_weight: float = 1.0
    # Number of points used to trace the efficient frontier.
    frontier_points: int = 60


PORTFOLIO = PortfolioConfig()


# Default primary ticker used by single-asset demonstrations (modeling phase).
PRIMARY_TICKER: str = "AAPL"
