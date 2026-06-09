"""
src/portfolio_optimization.py  —  CRISP-DM PHASE 5 (Portfolio Optimization)
===========================================================================
A self-contained Modern Portfolio Theory (MPT / Markowitz) engine built on
explicit matrix mathematics and ``scipy.optimize`` — no PyPortfolioOpt
dependency, so every formula is visible and auditable.

Core quantities (annualised)
----------------------------
Given a weight vector ``w`` (∑w = 1), expected return vector ``μ`` and
covariance matrix ``Σ`` of the assets' returns:

    portfolio return      μ_p = wᵀ μ
    portfolio variance    σ²_p = wᵀ Σ w
    portfolio volatility   σ_p = sqrt(σ²_p)
    Sharpe ratio           S   = (μ_p − r_f) / σ_p

Optimisation problems solved here
---------------------------------
  * **Maximum Sharpe (tangency) portfolio** — maximise S  s.t. ∑w = 1, bounds.
    Implemented as minimisation of the negative Sharpe ratio.
  * **Global Minimum Variance portfolio** — minimise σ²_p  s.t. ∑w = 1, bounds.
  * **Efficient Frontier** — for a grid of target returns, minimise variance
    s.t. μ_p = target, ∑w = 1, bounds. The locus of these points is the
    frontier; max-Sharpe and min-variance both lie on it.

Estimation inputs come from realised history (mean log returns, sample
covariance). Hooks are provided to override μ with model-implied expected
returns from Phase 4 if desired.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config
from src.data_ingestion import get_price_panel
from src.utils import banner, get_logger

logger = get_logger("portfolio_optimization")

TRADING_DAYS = config.TRADING_DAYS_PER_YEAR


# ===========================================================================
# Estimation of inputs (μ, Σ)
# ===========================================================================
def compute_returns(long: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Daily log-return panel for the investable universe (benchmark excluded)."""
    panel = get_price_panel(long, "Adj Close")[tickers].dropna()
    returns = np.log(panel / panel.shift(1)).dropna()
    return returns


def annualised_mean(returns: pd.DataFrame) -> pd.Series:
    """Annualised expected return per asset:  μ = mean_daily × 252."""
    return returns.mean() * TRADING_DAYS


def annualised_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised covariance matrix:  Σ = cov_daily × 252."""
    return returns.cov() * TRADING_DAYS


# ===========================================================================
# Portfolio algebra
# ===========================================================================
@dataclass
class PortfolioStats:
    weights: pd.Series
    exp_return: float
    volatility: float
    sharpe: float

    def as_dict(self) -> dict:
        return {
            "exp_return": self.exp_return,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "weights": self.weights.round(4).to_dict(),
        }


def portfolio_performance(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free: float = config.RISK_FREE_RATE,
) -> tuple[float, float, float]:
    """
    Return (expected_return, volatility, sharpe) for a weight vector.

    All inputs are annualised. This is the single source of truth used by both
    the objective functions and the reporting layer.
    """
    exp_return = float(weights @ mu)
    variance = float(weights @ cov @ weights)
    volatility = float(np.sqrt(max(variance, 1e-18)))
    sharpe = (exp_return - risk_free) / volatility
    return exp_return, volatility, sharpe


def _bounds(n_assets: int) -> tuple[tuple[float, float], ...]:
    """Per-asset weight bounds, controlled by the PortfolioConfig."""
    lo = -config.PORTFOLIO.max_weight if config.PORTFOLIO.allow_short else config.PORTFOLIO.min_weight
    hi = config.PORTFOLIO.max_weight
    return tuple((lo, hi) for _ in range(n_assets))


def _sum_to_one_constraint() -> dict:
    """Equality constraint enforcing a fully-invested portfolio (∑w = 1)."""
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


# ===========================================================================
# Optimisers
# ===========================================================================
def _solve(objective, n_assets: int, extra_constraints: list[dict] | None = None):
    """Shared SLSQP solver harness with an equal-weight warm start."""
    constraints = [_sum_to_one_constraint()]
    if extra_constraints:
        constraints.extend(extra_constraints)
    x0 = np.repeat(1.0 / n_assets, n_assets)  # equal-weight starting guess
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=_bounds(n_assets),
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        logger.warning("Optimizer did not converge cleanly: %s", result.message)
    return result.x


def maximum_sharpe_portfolio(
    mu: pd.Series, cov: pd.DataFrame, risk_free: float = config.RISK_FREE_RATE
) -> PortfolioStats:
    """
    Tangency portfolio — the long-only weights that maximise the Sharpe ratio.

    We minimise the negative Sharpe ratio because SciPy only minimises.
    """
    mu_v, cov_v = mu.to_numpy(), cov.to_numpy()

    def neg_sharpe(w: np.ndarray) -> float:
        _, _, s = portfolio_performance(w, mu_v, cov_v, risk_free)
        return -s

    w = _solve(neg_sharpe, len(mu))
    r, vol, s = portfolio_performance(w, mu_v, cov_v, risk_free)
    return PortfolioStats(pd.Series(w, index=mu.index), r, vol, s)


def minimum_variance_portfolio(
    mu: pd.Series, cov: pd.DataFrame, risk_free: float = config.RISK_FREE_RATE
) -> PortfolioStats:
    """Global minimum-variance portfolio — minimise σ²_p subject to ∑w = 1."""
    mu_v, cov_v = mu.to_numpy(), cov.to_numpy()

    def variance(w: np.ndarray) -> float:
        return float(w @ cov_v @ w)

    w = _solve(variance, len(mu))
    r, vol, s = portfolio_performance(w, mu_v, cov_v, risk_free)
    return PortfolioStats(pd.Series(w, index=mu.index), r, vol, s)


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = config.PORTFOLIO.frontier_points,
    risk_free: float = config.RISK_FREE_RATE,
) -> pd.DataFrame:
    """
    Trace the efficient frontier.

    For each target return τ in a grid spanning [min μ, max μ], solve:
        minimise   wᵀ Σ w
        subject to wᵀ μ = τ,  ∑w = 1,  bounds.

    Returns a DataFrame with one row per frontier point: target return,
    achieved volatility, Sharpe, and the full weight vector.
    """
    mu_v, cov_v = mu.to_numpy(), cov.to_numpy()
    n = len(mu)
    targets = np.linspace(mu_v.min(), mu_v.max(), n_points)

    rows = []
    for tau in targets:
        constraints = [
            {"type": "eq", "fun": lambda w, t=tau: float(w @ mu_v) - t},
        ]

        def variance(w: np.ndarray) -> float:
            return float(w @ cov_v @ w)

        w = _solve(variance, n, extra_constraints=constraints)
        r, vol, s = portfolio_performance(w, mu_v, cov_v, risk_free)
        row = {"target_return": tau, "exp_return": r, "volatility": vol, "sharpe": s}
        row.update({f"w_{name}": wi for name, wi in zip(mu.index, w)})
        rows.append(row)

    frontier = pd.DataFrame(rows)
    return frontier


# ===========================================================================
# Visualisation
# ===========================================================================
def plot_efficient_frontier(
    frontier: pd.DataFrame,
    max_sharpe: PortfolioStats,
    min_var: PortfolioStats,
    returns: pd.DataFrame | None = None,
    filename: str = "efficient_frontier.png",
) -> str:
    """
    Plot the efficient frontier with the two special portfolios highlighted,
    optionally overlaying each individual asset's risk/return point.
    """
    config.ensure_dirs()
    fig, ax = plt.subplots(figsize=(9, 6))

    # Frontier coloured by Sharpe ratio.
    sc = ax.scatter(
        frontier["volatility"],
        frontier["exp_return"],
        c=frontier["sharpe"],
        cmap="viridis",
        s=18,
        label="Efficient frontier",
    )
    fig.colorbar(sc, label="Sharpe ratio")

    # Individual assets.
    if returns is not None:
        mu = annualised_mean(returns)
        vol = returns.std() * np.sqrt(TRADING_DAYS)
        ax.scatter(vol, mu, marker="o", color="grey", s=60, edgecolor="black")
        for name in mu.index:
            ax.annotate(name, (vol[name], mu[name]), fontsize=9,
                        xytext=(5, 4), textcoords="offset points")

    # Special portfolios.
    ax.scatter(
        max_sharpe.volatility, max_sharpe.exp_return,
        marker="*", color="red", s=320, label="Max Sharpe", zorder=5,
    )
    ax.scatter(
        min_var.volatility, min_var.exp_return,
        marker="P", color="blue", s=180, label="Min Variance", zorder=5,
    )

    ax.set_xlabel("Annualised volatility (σ)")
    ax.set_ylabel("Annualised expected return (μ)")
    ax.set_title("Efficient Frontier — Mean-Variance Optimization")
    ax.legend()
    fig.tight_layout()
    path = config.FIGURE_DIR / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Saved efficient frontier → %s", path)
    return str(path)


# ===========================================================================
# Orchestrator
# ===========================================================================
@dataclass
class OptimizationResult:
    max_sharpe: PortfolioStats
    min_variance: PortfolioStats
    frontier: pd.DataFrame
    figure_path: str
    expected_returns: pd.Series
    cov_matrix: pd.DataFrame


def optimize_portfolio(
    long: pd.DataFrame,
    tickers: list[str] | None = None,
    expected_returns_override: pd.Series | None = None,
) -> OptimizationResult:
    """
    Full Phase-5 portfolio optimization run.

    Parameters
    ----------
    long : the tidy OHLCV frame.
    tickers : investable universe (defaults to config.TICKERS, excludes SPY).
    expected_returns_override : optionally substitute model-implied expected
        returns (e.g. annualised mean of Phase-4 predicted next-day returns)
        for the historical mean. Σ is always estimated from realised history.
    """
    tickers = tickers or config.TICKERS
    logger.info(banner("PHASE 5 — PORTFOLIO OPTIMIZATION (MPT)"))

    returns = compute_returns(long, tickers)
    mu = (
        expected_returns_override
        if expected_returns_override is not None
        else annualised_mean(returns)
    )
    mu = mu.reindex(tickers)
    cov = annualised_cov(returns)

    max_sharpe = maximum_sharpe_portfolio(mu, cov)
    min_var = minimum_variance_portfolio(mu, cov)
    frontier = efficient_frontier(mu, cov)
    fig_path = plot_efficient_frontier(frontier, max_sharpe, min_var, returns)

    # ---- console report ----------------------------------------------------
    print(banner("Maximum Sharpe Ratio Portfolio"))
    print(f"  Expected return : {max_sharpe.exp_return:7.2%}")
    print(f"  Volatility      : {max_sharpe.volatility:7.2%}")
    print(f"  Sharpe ratio    : {max_sharpe.sharpe:7.2f}")
    print("  Weights:")
    for t, w in max_sharpe.weights.items():
        print(f"      {t:<6} {w:7.2%}")

    print(banner("Minimum Variance Portfolio"))
    print(f"  Expected return : {min_var.exp_return:7.2%}")
    print(f"  Volatility      : {min_var.volatility:7.2%}")
    print(f"  Sharpe ratio    : {min_var.sharpe:7.2f}")
    print("  Weights:")
    for t, w in min_var.weights.items():
        print(f"      {t:<6} {w:7.2%}")

    # ---- persist weights ----------------------------------------------------
    config.ensure_dirs()
    weights_df = pd.DataFrame(
        {
            "max_sharpe": max_sharpe.weights,
            "min_variance": min_var.weights,
        }
    )
    weights_df.to_csv(config.REPORT_DIR / "portfolio_weights.csv")
    frontier.to_csv(config.REPORT_DIR / "efficient_frontier.csv", index=False)

    return OptimizationResult(
        max_sharpe=max_sharpe,
        min_variance=min_var,
        frontier=frontier,
        figure_path=fig_path,
        expected_returns=mu,
        cov_matrix=cov,
    )


# ===========================================================================
# STEP 2 — DYNAMIC ROLLING REBALANCING ENGINE
# ===========================================================================
# The static optimizer above produces a single snapshot allocation from the
# full sample (which itself peeks at the whole history). A realistic process
# re-optimises periodically using ONLY past data, holds the weights between
# rebalances, and pays a transaction-cost drag on turnover. The engine below
# simulates exactly that and produces a compounding, out-of-sample equity
# curve. Every estimation window is sliced with ``.loc[:date]`` so look-ahead
# bias is structurally impossible.


@dataclass
class RollingBacktestResult:
    """Output of a walk-forward portfolio backtest."""

    strategy: str
    equity_curve: pd.Series          # net of costs, base = 1.0
    gross_equity_curve: pd.Series    # before costs
    daily_returns: pd.Series         # net daily simple returns
    weights_history: pd.DataFrame    # index = rebalance date, columns = tickers
    turnover: pd.Series              # one-way turnover per rebalance date
    total_cost: float                # cumulative cost as a return-fraction sum
    rebalance_dates: pd.DatetimeIndex
    n_rebalances: int


def _rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """
    Last available trading day of each calendar month ('M') or quarter ('Q').

    Decisions are made at the close of these dates and take effect on the next
    trading day, so no future information leaks into the chosen weights.
    """
    freq = frequency.upper()
    if freq not in {"M", "Q"}:
        raise ValueError("frequency must be 'M' (monthly) or 'Q' (quarterly).")
    periods = index.to_period(freq)
    df = pd.DataFrame({"date": index, "period": periods})
    last_per_period = df.groupby("period", sort=True)["date"].max()
    return pd.DatetimeIndex(last_per_period.values)


def _optimise_weights(
    strategy: str,
    mu: pd.Series,
    cov: pd.DataFrame,
) -> pd.Series:
    """Dispatch to the requested allocation rule and return a weight Series."""
    if strategy == "max_sharpe":
        return maximum_sharpe_portfolio(mu, cov).weights
    if strategy == "min_variance":
        return minimum_variance_portfolio(mu, cov).weights
    if strategy == "equal_weight":
        n = len(mu)
        return pd.Series(np.repeat(1.0 / n, n), index=mu.index)
    raise ValueError(f"Unknown strategy '{strategy}'.")


def rolling_rebalance_backtest(
    long: pd.DataFrame,
    tickers: list[str] | None = None,
    strategy: str = config.REBALANCE.strategy,
    frequency: str = config.REBALANCE.frequency,
    lookback_days: int = config.REBALANCE.lookback_days,
    rolling_lookback: bool = config.REBALANCE.rolling_lookback,
    cost_rate: float = config.REBALANCE.cost_rate,
    mu_override: pd.Series | None = None,
    mu_provider=None,
    weight_transform=None,
) -> RollingBacktestResult:
    """
    Walk-forward portfolio backtest with periodic rebalancing + trading costs.

    Procedure
    ---------
    1. Build the daily SIMPLE-return panel (simple returns aggregate linearly
       across assets and compound across time — the correct unit for an equity
       curve).
    2. Determine rebalance dates (month/quarter ends).
    3. At each rebalance date ``d`` use only returns in ``(:d]`` (optionally the
       trailing ``lookback_days``) to estimate μ (annualised mean of log
       returns, or ``mu_override``) and Σ (annualised covariance), then solve
       for the target weights under ``strategy``.
    4. Weights take effect the NEXT trading day and then DRIFT with returns
       until the following rebalance. At each rebalance we charge
       ``turnover × cost_rate`` as a same-day return drag, where turnover is
       ``Σ|w_target − w_drifted|`` (full traded volume, buys + sells).

    Signal hooks (Step: signal-driven allocation)
    ---------------------------------------------
    ``mu_provider`` : optional ``callable(date, window_log_returns) -> Series``.
        When supplied it supersedes the historical mean / ``mu_override`` for
        the expected-return vector at each rebalance. Returning ``None`` (or a
        vector with NaNs) falls back to the historical mean. This is how a
        model-derived expected-return signal is injected WITHOUT look-ahead —
        the provider only ever receives data up to ``date``.
    ``weight_transform`` : optional ``callable(date, base_weights, window) ->
        Series``. Applied to the optimiser's output to tilt the allocation
        (e.g. overweight high-signal names) before it is held.

    Returns a :class:`RollingBacktestResult` with the net & gross equity
    curves, the full weight history, and turnover/cost diagnostics.
    """
    tickers = tickers or config.TICKERS
    panel = get_price_panel(long, "Adj Close")[tickers].dropna()
    simple = panel.pct_change().dropna()
    log_ret = np.log(panel / panel.shift(1)).dropna()
    all_dates = simple.index

    rebal_dates = _rebalance_dates(all_dates, frequency)
    min_obs = lookback_days if rolling_lookback else max(len(tickers) + 2, 21)

    # ---- 1. Pre-compute target weights at each valid rebalance date --------
    targets: dict[pd.Timestamp, pd.Series] = {}
    turnover_records: dict[pd.Timestamp, float] = {}
    for d in rebal_dates:
        window = log_ret.loc[:d]
        if rolling_lookback:
            window = window.tail(lookback_days)
        if len(window) < min_obs:
            continue

        # Expected-return vector μ — provider (signal) > static override > history.
        mu = None
        if mu_provider is not None:
            mu = mu_provider(d, window)
        elif mu_override is not None:
            mu = mu_override.reindex(tickers)
        if mu is None or pd.Series(mu).reindex(tickers).isna().any():
            mu = annualised_mean(window)          # robust fallback
        else:
            mu = pd.Series(mu).reindex(tickers)

        cov = annualised_cov(window)
        w = _optimise_weights(strategy, mu, cov).reindex(tickers).fillna(0.0)
        if weight_transform is not None:
            w = pd.Series(weight_transform(d, w, window)).reindex(tickers).fillna(0.0)
        targets[d] = w

    if not targets:
        raise RuntimeError(
            "No rebalance date had enough history. Lower lookback_days or "
            "widen the date range."
        )

    # Map each target to the first trading day it becomes active.
    eff_target: dict[pd.Timestamp, pd.Series] = {}
    pos_of = {dt: i for i, dt in enumerate(all_dates)}
    for d, w in targets.items():
        nxt = pos_of[d] + 1
        if nxt < len(all_dates):
            eff_target[all_dates[nxt]] = w

    # ---- 2. Simulate forward day-by-day ------------------------------------
    sim_dates = all_dates[all_dates >= min(eff_target)]
    equity, gross_equity = 1.0, 1.0
    drift_w: pd.Series | None = None
    rows = []
    total_cost = 0.0

    for date in sim_dates:
        cost = 0.0
        if date in eff_target:
            new_w = eff_target[date]
            if drift_w is None:
                turnover = float(new_w.abs().sum())            # initial buy from cash
            else:
                turnover = float((new_w - drift_w.reindex(new_w.index).fillna(0.0)).abs().sum())
            cost = turnover * cost_rate
            total_cost += cost
            turnover_records[date] = turnover
            drift_w = new_w.copy()

        r = simple.loc[date, drift_w.index]
        gross_ret = float((drift_w * r).sum())
        net_ret = gross_ret - cost

        gross_equity *= 1.0 + gross_ret
        equity *= 1.0 + net_ret
        rows.append(
            {"date": date, "net_ret": net_ret, "gross_ret": gross_ret,
             "equity": equity, "gross_equity": gross_equity}
        )

        # Drift the weights with the day's price moves for tomorrow.
        drifted = drift_w * (1.0 + r)
        drift_w = drifted / drifted.sum()

    sim = pd.DataFrame(rows).set_index("date")
    weights_history = pd.DataFrame(targets).T
    weights_history.index.name = "rebalance_date"

    result = RollingBacktestResult(
        strategy=strategy,
        equity_curve=sim["equity"],
        gross_equity_curve=sim["gross_equity"],
        daily_returns=sim["net_ret"],
        weights_history=weights_history,
        turnover=pd.Series(turnover_records).sort_index(),
        total_cost=total_cost,
        rebalance_dates=pd.DatetimeIndex(list(targets.keys())),
        n_rebalances=len(targets),
    )
    logger.info(
        "[%s] rolling backtest: %d rebalances, %s→%s, total cost drag=%.2f%%, "
        "final equity=%.3fx (gross %.3fx)",
        strategy,
        result.n_rebalances,
        sim.index.min().date(),
        sim.index.max().date(),
        total_cost * 100,
        equity,
        gross_equity,
    )
    return result
