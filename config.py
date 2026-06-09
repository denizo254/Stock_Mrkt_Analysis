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
    # Reproducibility.
    random_state: int = 42
    # Which engine to use. "xgboost" or "linear" (Ridge / LogisticRegression).
    regressor: str = "xgboost"
    classifier: str = "xgboost"

    # --- Anti-overfitting structural defaults (Step 1) ---------------------
    # Aggressive row/column subsampling forces the trees to diversify across
    # features and decorrelates the ensemble — the single most effective lever
    # against fitting market microstructure noise. These are FIXED on the
    # estimator; the regularisation penalties below are what we TUNE.
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    n_estimators: int = 300

    # --- Hyper-parameter search grids (heavily regularised) ----------------
    # Shallow trees + L1 (alpha) + L2 (lambda) + split-gain penalty (gamma>0).
    # Kept deliberately compact so a walk-forward grid search stays tractable.
    max_depth_grid: tuple[int, ...] = (1, 2, 3)
    learning_rate_grid: tuple[float, ...] = (0.01, 0.05)
    reg_alpha_grid: tuple[float, ...] = (0.0, 1.0)        # L1
    reg_lambda_grid: tuple[float, ...] = (1.0, 10.0)      # L2
    gamma_grid: tuple[float, ...] = (0.1, 1.0)            # min split loss > 0


MODEL = ModelConfig()


# ---------------------------------------------------------------------------
# Walk-Forward Validation (Phase 4, Step 1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WalkForwardConfig:
    """
    Rolling walk-forward cross-validation geometry.

    Windows are expressed in calendar months and converted to a sample count
    via ``trading_days_per_month``. A 12-month train / 1-month test window that
    rolls forward one month at a time emulates how a desk actually re-fits a
    model in production.
    """

    train_months: int = 12
    test_months: int = 1
    step_months: int = 1
    trading_days_per_month: int = 21      # ~252 / 12
    rolling: bool = True                  # True = fixed rolling window; False = anchored/expanding
    # During hyper-parameter search a full step-by-step WFV can produce dozens
    # of folds (≈ one per month of history), which makes the grid search slow.
    # Cap the search to the most-recent N folds (the regime that matters most).
    cv_max_splits: int = 8

    @property
    def train_size(self) -> int:
        return self.train_months * self.trading_days_per_month

    @property
    def test_size(self) -> int:
        return self.test_months * self.trading_days_per_month

    @property
    def step_size(self) -> int:
        return self.step_months * self.trading_days_per_month


WALK_FORWARD = WalkForwardConfig()


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
    # Covariance estimator: 'sample' (empirical) or 'ledoit_wolf' (shrinkage).
    # Ledoit-Wolf shrinks the noisy sample covariance toward a structured
    # target, which materially stabilises out-of-sample portfolio weights.
    cov_method: str = "sample"


PORTFOLIO = PortfolioConfig()


# ---------------------------------------------------------------------------
# Risk overlays (robustness layer over the rolling backtest)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskOverlayConfig:
    """
    Optional risk controls applied on top of the optimiser's raw weights.
    Every field defaults to "off" so the overlays are strictly opt-in and the
    baseline behaviour is unchanged.
    """

    # Per-name position cap (e.g. 0.35 => no asset above 35%). None = no cap
    # beyond the optimiser's own [min_weight, max_weight] bounds.
    max_weight: float | None = None
    # Annualised volatility target. When set, exposure is scaled so the
    # portfolio's ex-ante vol ≈ target; the remainder sits in cash at the
    # risk-free rate. None = fully invested (exposure 1.0).
    target_vol: float | None = None
    max_leverage: float = 1.0          # cap on exposure when vol-targeting
    # Drawdown stop: de-risk to cash once the equity curve falls this far below
    # its peak (e.g. 0.25 => -25%); re-risk once recovered to within
    # ``drawdown_reenter`` of the peak. None = no stop.
    drawdown_stop: float | None = None
    drawdown_reenter: float = 0.10


RISK = RiskOverlayConfig()


# ---------------------------------------------------------------------------
# Dynamic rolling rebalancing engine (Phase 5, Step 2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RebalanceConfig:
    """
    Configuration for the walk-forward portfolio backtest.

    At each rebalancing date the optimiser re-estimates μ and Σ using ONLY the
    trailing ``lookback_days`` of returns (strictly before the rebalance date),
    then holds the resulting weights until the next rebalance. Turnover at each
    rebalance is charged ``transaction_cost_bps`` basis points.
    """

    frequency: str = "M"               # 'M' = month-end, 'Q' = quarter-end
    lookback_days: int = 252           # trailing estimation window (1 trading year)
    rolling_lookback: bool = True      # True = fixed window; False = expanding/anchored
    transaction_cost_bps: float = 7.0  # 5–10 bps charged per unit of turnover
    strategy: str = "max_sharpe"       # 'max_sharpe' | 'min_variance' | 'equal_weight'

    @property
    def cost_rate(self) -> float:
        """Transaction cost as a decimal fraction (e.g. 7 bps -> 0.0007)."""
        return self.transaction_cost_bps / 10_000.0


REBALANCE = RebalanceConfig()


# ---------------------------------------------------------------------------
# Explainability / SHAP (Phase 5, Step 3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExplainConfig:
    enable_shap: bool = True           # fall back to native gain if shap missing
    shap_sample_size: int = 300        # rows sampled from the test set for SHAP
    top_n_features: int = 20           # how many features to render in plots
    random_state: int = 42


EXPLAIN = ExplainConfig()


# Default primary ticker used by single-asset demonstrations (modeling phase).
PRIMARY_TICKER: str = "AAPL"
