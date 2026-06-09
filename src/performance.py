"""
src/performance.py  —  CRISP-DM PHASE 5 (Evaluation: portfolio performance)
===========================================================================
Risk-adjusted performance analytics that translate a weight vector into the
investor-facing scorecard defined in Phase 1, benchmarked against the S&P 500
(SPY).

Metrics
-------
Sharpe ratio       (μ − r_f) / σ                          — return per unit total risk
Sortino ratio      (μ − r_f) / σ_downside                 — return per unit *downside* risk
Maximum Drawdown   max peak-to-trough decline of equity   — worst-case pain
Calmar ratio       annualised return / |max drawdown|     — return per unit drawdown
Annualised return  geometric, from the daily equity curve
Annualised vol     daily σ × √252

Each metric is computed for the candidate portfolio AND for SPY so the
out-performance (or lack thereof) is explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

import config
from src.data_ingestion import get_price_panel
from src.utils import banner, get_logger

logger = get_logger("performance")

TD = config.TRADING_DAYS_PER_YEAR
RF = config.RISK_FREE_RATE


# ===========================================================================
# Building a portfolio return stream
# ===========================================================================
def portfolio_return_series(
    long: pd.DataFrame, weights: pd.Series
) -> pd.Series:
    """
    Daily simple-return series of a fixed-weight portfolio.

    We use simple (arithmetic) returns here because portfolio returns are a
    linear combination of constituent simple returns — that identity does not
    hold for log returns. Weights are assumed constant (a daily-rebalanced
    portfolio), which is the standard MPT interpretation.
    """
    panel = get_price_panel(long, "Adj Close")[list(weights.index)].dropna()
    simple = panel.pct_change().dropna()
    return simple @ weights.reindex(panel.columns).to_numpy()


def benchmark_return_series(long: pd.DataFrame) -> pd.Series:
    """Daily simple-return series of the SPY benchmark."""
    panel = get_price_panel(long, "Adj Close")
    spy = panel[config.BENCHMARK].dropna()
    return spy.pct_change().dropna()


# ===========================================================================
# Metric primitives
# ===========================================================================
def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Cumulative growth of 1 unit:  ∏(1 + r_t)."""
    return initial * (1.0 + returns).cumprod()


def annualised_return(returns: pd.Series) -> float:
    """Geometric annualised return derived from the realised equity curve."""
    total_growth = float((1.0 + returns).prod())
    years = len(returns) / TD
    if years <= 0:
        return 0.0
    return total_growth ** (1.0 / years) - 1.0


def annualised_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=0) * np.sqrt(TD))


def sharpe_ratio(returns: pd.Series, risk_free: float = RF) -> float:
    """Annualised Sharpe ratio using a daily-decomposed risk-free rate."""
    excess = returns - risk_free / TD
    denom = excess.std(ddof=0)
    if denom == 0:
        return 0.0
    return float(excess.mean() / denom * np.sqrt(TD))


def sortino_ratio(returns: pd.Series, risk_free: float = RF) -> float:
    """
    Annualised Sortino ratio — like Sharpe but penalises only downside
    deviation (returns below the risk-free target).
    """
    excess = returns - risk_free / TD
    downside = excess[excess < 0]
    downside_dev = np.sqrt((downside ** 2).mean()) if len(downside) else 0.0
    if downside_dev == 0:
        return 0.0
    return float(excess.mean() / downside_dev * np.sqrt(TD))


def maximum_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough decline of the equity curve (a negative number).

        DD_t = equity_t / running_max(equity)_t − 1
        MDD  = min_t DD_t
    """
    curve = equity_curve(returns)
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series) -> float:
    """Annualised return divided by the absolute maximum drawdown."""
    mdd = maximum_drawdown(returns)
    if mdd == 0:
        return 0.0
    return float(annualised_return(returns) / abs(mdd))


# ===========================================================================
# Scorecard
# ===========================================================================
@dataclass
class PerformanceScorecard:
    label: str
    ann_return: float
    ann_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float


def score(returns: pd.Series, label: str) -> PerformanceScorecard:
    """Compute the full scorecard for one return stream."""
    return PerformanceScorecard(
        label=label,
        ann_return=annualised_return(returns),
        ann_volatility=annualised_volatility(returns),
        sharpe=sharpe_ratio(returns),
        sortino=sortino_ratio(returns),
        max_drawdown=maximum_drawdown(returns),
        calmar=calmar_ratio(returns),
    )


def benchmark_report(
    long: pd.DataFrame,
    named_weights: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Compare one or more candidate portfolios against the SPY benchmark.

    Parameters
    ----------
    named_weights : mapping of portfolio label → weight Series, e.g.
        {"Max Sharpe": max_sharpe.weights, "Min Variance": min_var.weights}.

    Returns a tidy scorecard DataFrame (rows = portfolios + SPY).
    """
    logger.info(banner("PHASE 5 — PORTFOLIO PERFORMANCE vs S&P 500"))

    cards: list[PerformanceScorecard] = []
    for label, weights in named_weights.items():
        ret = portfolio_return_series(long, weights)
        cards.append(score(ret, label))

    cards.append(score(benchmark_return_series(long), f"{config.BENCHMARK} (benchmark)"))

    report = pd.DataFrame([asdict(c) for c in cards]).set_index("label")

    print(banner("Risk-adjusted performance scorecard"))
    pretty = report.copy()
    for col in ["ann_return", "ann_volatility", "max_drawdown"]:
        pretty[col] = pretty[col].map(lambda x: f"{x:6.2%}")
    for col in ["sharpe", "sortino", "calmar"]:
        pretty[col] = pretty[col].map(lambda x: f"{x:6.2f}")
    print(pretty.to_string())

    config.ensure_dirs()
    report.to_csv(config.REPORT_DIR / "portfolio_performance.csv")
    logger.info("Saved performance scorecard → %s", config.REPORT_DIR / "portfolio_performance.csv")
    return report


# ===========================================================================
# STEP 2 — Dynamic (rolling backtest) performance vs benchmark
# ===========================================================================
def _spy_returns_on(index: pd.DatetimeIndex, long: pd.DataFrame) -> pd.Series:
    """SPY daily simple returns reindexed onto the backtest's date range."""
    spy = benchmark_return_series(long)
    return spy.reindex(index).dropna()


def dynamic_benchmark_report(
    long: pd.DataFrame,
    backtests: dict[str, "object"],
) -> pd.DataFrame:
    """
    Score one or more rolling-backtest equity curves against SPY over the
    SAME out-of-sample window.

    Parameters
    ----------
    backtests : mapping label → ``RollingBacktestResult`` (kept loosely typed
        to avoid a circular import with ``portfolio_optimization``).

    Returns a scorecard DataFrame (rows = strategies + SPY). The strategy
    metrics are net of transaction costs, so this is a like-for-like,
    cost-aware comparison.
    """
    logger.info(banner("PHASE 5 — DYNAMIC ROLLING PORTFOLIO vs S&P 500"))

    cards: list[PerformanceScorecard] = []
    # Use the union of backtest dates to align the benchmark window.
    all_index = None
    for label, bt in backtests.items():
        rets = bt.daily_returns
        cards.append(score(rets, f"{label} (net)"))
        all_index = rets.index if all_index is None else all_index.union(rets.index)

    spy_rets = _spy_returns_on(all_index, long)
    cards.append(score(spy_rets, f"{config.BENCHMARK} (benchmark)"))

    report = pd.DataFrame([asdict(c) for c in cards]).set_index("label")

    print(banner("Dynamic risk-adjusted scorecard (net of costs)"))
    pretty = report.copy()
    for col in ["ann_return", "ann_volatility", "max_drawdown"]:
        pretty[col] = pretty[col].map(lambda x: f"{x:6.2%}")
    for col in ["sharpe", "sortino", "calmar"]:
        pretty[col] = pretty[col].map(lambda x: f"{x:6.2f}")
    print(pretty.to_string())

    config.ensure_dirs()
    report.to_csv(config.REPORT_DIR / "dynamic_portfolio_performance.csv")
    logger.info(
        "Saved dynamic scorecard → %s",
        config.REPORT_DIR / "dynamic_portfolio_performance.csv",
    )
    return report


def plot_dynamic_equity_curves(
    long: pd.DataFrame,
    backtests: dict[str, "object"],
    filename: str = "dynamic_equity_curve.png",
) -> str:
    """
    Plot the net compounding equity curve of each strategy against a
    buy-and-hold SPY curve over the matching window (base = 1.0).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.ensure_dirs()
    fig, ax = plt.subplots(figsize=(11, 6))

    union_index = None
    for label, bt in backtests.items():
        curve = bt.equity_curve
        ax.plot(curve.index, curve.values, linewidth=1.6, label=f"{label} (net)")
        union_index = curve.index if union_index is None else union_index.union(curve.index)

    spy_rets = _spy_returns_on(union_index, long)
    spy_curve = equity_curve(spy_rets)
    ax.plot(spy_curve.index, spy_curve.values, linewidth=1.6,
            linestyle="--", color="black", label=f"{config.BENCHMARK} (buy & hold)")

    ax.set_title("Dynamic Rebalanced Portfolio vs S&P 500 — Growth of $1 (net of costs)")
    ax.set_ylabel("Equity (base = 1.0)")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    path = config.FIGURE_DIR / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Saved dynamic equity curve → %s", path)
    return str(path)
