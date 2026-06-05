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
