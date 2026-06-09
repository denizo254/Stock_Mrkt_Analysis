"""
app.py  —  CRISP-DM PHASE 6 (Interactive Presentation Layer)
============================================================
A lightweight Streamlit dashboard that sits on top of the core pipeline
modules (it imports and reuses them — it does NOT re-implement any analytics).

Run with:
    streamlit run app.py

Panels
------
* **Sidebar** — asset-universe multiselect, benchmark (SPY), rebalancing
  frequency, estimation lookback, transaction-cost slider, optimisation
  strategy, and the model-implied-μ toggle.
* **Dynamic Equity Curve** — net-of-cost compounding growth of $1 for the
  rolling-rebalanced portfolio versus a buy-and-hold S&P 500.
* **Rolling Weight Allocations** — how the optimiser re-allocates capital at
  each rebalance, as a stacked area over time.
* **Efficient Frontier** — the full mean-variance frontier with the Max-Sharpe
  and Min-Variance portfolios marked.
* **Scorecards** — risk-adjusted metrics (Sharpe / Sortino / Drawdown / Calmar)
  and the latest target weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src.data_ingestion import download_market_data
from src.performance import (
    annualised_return,
    annualised_volatility,
    benchmark_return_series,
    equity_curve,
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
)
from src.portfolio_optimization import (
    optimize_portfolio,
    rolling_rebalance_backtest,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Market Analysis & Investment Optimization",
    page_icon="📈",
    layout="wide",
)

DEFAULT_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "TSLA", "JPM"]


# ---------------------------------------------------------------------------
# Cached data + compute layers (keyed on their inputs)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Downloading market data …")
def load_data(symbols: tuple[str, ...], start: str, end: str | None) -> pd.DataFrame:
    """Cached yfinance pull. Tuple args make the cache key hashable."""
    return download_market_data(
        symbols=list(symbols), start=start, end=end, force_refresh=False
    )


@st.cache_data(show_spinner="Running rolling rebalancing backtest …")
def run_backtest(
    symbols: tuple[str, ...],
    tickers: tuple[str, ...],
    start: str,
    end: str | None,
    strategy: str,
    frequency: str,
    lookback_days: int,
    cost_bps: float,
    cov_method: str = "sample",
    max_weight: float | None = None,
    target_vol: float | None = None,
    drawdown_stop: float | None = None,
):
    """Cached rolling backtest; returns a plain dict so it is cache-friendly."""
    long = load_data(symbols, start, end)
    bt = rolling_rebalance_backtest(
        long,
        tickers=list(tickers),
        strategy=strategy,
        frequency=frequency,
        lookback_days=lookback_days,
        cost_rate=cost_bps / 10_000.0,
        cov_method=cov_method,
        max_weight=max_weight,
        target_vol=target_vol,
        drawdown_stop=drawdown_stop,
    )
    return {
        "equity_curve": bt.equity_curve,
        "daily_returns": bt.daily_returns,
        "weights_history": bt.weights_history,
        "turnover": bt.turnover,
        "total_cost": bt.total_cost,
        "n_rebalances": bt.n_rebalances,
        "exposure": bt.exposure,
        "n_stops": bt.n_stops,
    }


@st.cache_data(show_spinner="Computing efficient frontier …")
def run_frontier(symbols: tuple[str, ...], tickers: tuple[str, ...], start: str, end: str | None):
    long = load_data(symbols, start, end)
    opt = optimize_portfolio(long, tickers=list(tickers))
    return {
        "frontier": opt.frontier,
        "max_sharpe": opt.max_sharpe,
        "min_variance": opt.min_variance,
    }


def scorecard(returns: pd.Series, label: str) -> dict:
    """Risk-adjusted metric row for a return stream."""
    return {
        "Strategy": label,
        "Ann. Return": annualised_return(returns),
        "Ann. Vol": annualised_volatility(returns),
        "Sharpe": sharpe_ratio(returns),
        "Sortino": sortino_ratio(returns),
        "Max Drawdown": maximum_drawdown(returns),
        "Calmar": calmar_ratio(returns),
    }


# ---------------------------------------------------------------------------
# Sidebar — user inputs
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

tickers = st.sidebar.multiselect(
    "Asset universe",
    options=DEFAULT_UNIVERSE,
    default=config.TICKERS,
    help="Equities to optimise over. SPY is always added as the benchmark.",
)

st.sidebar.text_input("Benchmark", value=config.BENCHMARK, disabled=True)

strategy_label = st.sidebar.selectbox(
    "Optimisation strategy",
    options=["Max Sharpe", "Min Variance", "Equal Weight"],
    index=0,
)
STRATEGY_MAP = {"Max Sharpe": "max_sharpe", "Min Variance": "min_variance", "Equal Weight": "equal_weight"}

freq_label = st.sidebar.selectbox("Rebalancing frequency", ["Monthly", "Quarterly"], index=0)
FREQ_MAP = {"Monthly": "M", "Quarterly": "Q"}

lookback = st.sidebar.slider("Estimation lookback (trading days)", 63, 504,
                             config.REBALANCE.lookback_days, step=21)
cost_bps = st.sidebar.slider("Transaction cost (basis points)", 0.0, 25.0,
                             config.REBALANCE.transaction_cost_bps, step=0.5)

start_date = st.sidebar.date_input("Start date", value=pd.Timestamp(config.START_DATE))

# --- Robustness overlays (opt-in; sliders at their min/max mean "off") ------
with st.sidebar.expander("🛡️ Robustness overlays"):
    use_shrinkage = st.checkbox(
        "Ledoit-Wolf covariance", value=False,
        help="Shrinkage estimator → stabler, less extreme weights.",
    )
    max_w = st.slider("Max weight per name (1.00 = off)", 0.20, 1.0, 1.0, 0.05)
    tgt_vol = st.slider("Target volatility (0 = off)", 0.0, 0.40, 0.0, 0.01)
    dd_stop = st.slider("Drawdown stop (0 = off)", 0.0, 0.50, 0.0, 0.05)

cov_method = "ledoit_wolf" if use_shrinkage else "sample"
max_weight = max_w if max_w < 1.0 else None
target_vol = tgt_vol if tgt_vol > 0 else None
drawdown_stop = dd_stop if dd_stop > 0 else None

run = st.sidebar.button("🚀 Run analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main canvas
# ---------------------------------------------------------------------------
st.title("📈 Stock Market Analysis & Investment Optimization")
st.caption(
    "CRISP-DM pipeline · walk-forward modeling · dynamic rolling MPT optimization "
    "· transaction-cost aware · SHAP explainability"
)

if not tickers:
    st.warning("Select at least two tickers in the sidebar to begin.")
    st.stop()

if not run:
    st.info("Configure the universe and parameters in the sidebar, then click **Run analysis**.")
    st.stop()

if len(tickers) < 2:
    st.error("Mean-variance optimization needs at least two assets.")
    st.stop()

symbols = tuple(dict.fromkeys(tickers + [config.BENCHMARK]))
tickers_t = tuple(tickers)
start_str = pd.Timestamp(start_date).strftime("%Y-%m-%d")
strategy = STRATEGY_MAP[strategy_label]
frequency = FREQ_MAP[freq_label]

long = load_data(symbols, start_str, None)

# --- run the selected strategy through the rolling backtest ----------------
bt_primary = run_backtest(symbols, tickers_t, start_str, None, strategy,
                          frequency, lookback, cost_bps,
                          cov_method=cov_method, max_weight=max_weight,
                          target_vol=target_vol, drawdown_stop=drawdown_stop)

_active = [n for n, on in [
    ("Ledoit-Wolf", use_shrinkage), (f"cap {max_w:.0%}", max_weight is not None),
    (f"vol-target {tgt_vol:.0%}", target_vol is not None),
    (f"DD-stop {dd_stop:.0%}", drawdown_stop is not None),
] if on]
if _active:
    st.info("🛡️ Active risk overlays: " + ", ".join(_active)
            + (f"  ·  drawdown stop triggered {bt_primary['n_stops']}×"
               if drawdown_stop is not None else ""))

# ===========================================================================
# Panel 1 — Dynamic equity curve vs SPY
# ===========================================================================
st.subheader("1 · Dynamic Equity Curve vs S&P 500 (net of costs)")

eq = bt_primary["equity_curve"]
spy_ret = benchmark_return_series(long).reindex(eq.index).dropna()
spy_curve = equity_curve(spy_ret)

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines",
                            name=f"{strategy_label} (net)", line=dict(width=2)))
fig_eq.add_trace(go.Scatter(x=spy_curve.index, y=spy_curve.values, mode="lines",
                            name="SPY (buy & hold)", line=dict(width=2, dash="dash", color="black")))
fig_eq.update_layout(height=420, yaxis_title="Growth of $1", xaxis_title="Date",
                     legend=dict(orientation="h", y=1.02, yanchor="bottom"),
                     margin=dict(t=20, b=10))
st.plotly_chart(fig_eq, use_container_width=True)

# KPI row
col1, col2, col3, col4 = st.columns(4)
port_card = scorecard(bt_primary["daily_returns"], strategy_label)
spy_card = scorecard(spy_ret, "SPY")
col1.metric("Portfolio Sharpe", f"{port_card['Sharpe']:.2f}", f"{port_card['Sharpe'] - spy_card['Sharpe']:+.2f} vs SPY")
col2.metric("Portfolio Ann. Return", f"{port_card['Ann. Return']:.1%}",
            f"{port_card['Ann. Return'] - spy_card['Ann. Return']:+.1%} vs SPY")
col3.metric("Max Drawdown", f"{port_card['Max Drawdown']:.1%}",
            f"{port_card['Max Drawdown'] - spy_card['Max Drawdown']:+.1%} vs SPY", delta_color="inverse")
col4.metric("Rebalances / Cost drag", f"{bt_primary['n_rebalances']}",
            f"{bt_primary['total_cost']:.2%} total")

# ===========================================================================
# Panel 2 — Rolling weight allocations
# ===========================================================================
st.subheader("2 · Rolling Asset Allocation Over Time")

w_hist = bt_primary["weights_history"]
fig_w = go.Figure()
for col in w_hist.columns:
    fig_w.add_trace(go.Scatter(
        x=w_hist.index, y=w_hist[col].values, mode="lines",
        name=col, stackgroup="one", line=dict(width=0.5, shape="hv"),
    ))
fig_w.update_layout(height=380, yaxis_title="Weight", xaxis_title="Rebalance date",
                    yaxis=dict(tickformat=".0%"), margin=dict(t=20, b=10),
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
st.plotly_chart(fig_w, use_container_width=True)

# ===========================================================================
# Panel 3 — Efficient frontier
# ===========================================================================
st.subheader("3 · Efficient Frontier (full-sample mean-variance)")

front = run_frontier(symbols, tickers_t, start_str, None)
frontier = front["frontier"]
ms, mv = front["max_sharpe"], front["min_variance"]

fig_f = go.Figure()
fig_f.add_trace(go.Scatter(
    x=frontier["volatility"], y=frontier["exp_return"], mode="markers",
    marker=dict(size=7, color=frontier["sharpe"], colorscale="Viridis",
                colorbar=dict(title="Sharpe"), showscale=True),
    name="Efficient frontier",
))
fig_f.add_trace(go.Scatter(x=[ms.volatility], y=[ms.exp_return], mode="markers",
                           marker=dict(symbol="star", size=20, color="red"), name="Max Sharpe"))
fig_f.add_trace(go.Scatter(x=[mv.volatility], y=[mv.exp_return], mode="markers",
                           marker=dict(symbol="x", size=16, color="blue"), name="Min Variance"))
fig_f.update_layout(height=460, xaxis_title="Annualised volatility (σ)",
                    yaxis_title="Annualised expected return (μ)",
                    xaxis=dict(tickformat=".0%"), yaxis=dict(tickformat=".0%"),
                    margin=dict(t=20, b=10))
st.plotly_chart(fig_f, use_container_width=True)

# ===========================================================================
# Panel 4 — Scorecards & weights
# ===========================================================================
st.subheader("4 · Risk-Adjusted Scorecard & Target Weights")
left, right = st.columns([3, 2])

with left:
    score_df = pd.DataFrame([port_card, spy_card]).set_index("Strategy")
    styled = score_df.style.format({
        "Ann. Return": "{:.2%}", "Ann. Vol": "{:.2%}", "Max Drawdown": "{:.2%}",
        "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Calmar": "{:.2f}",
    })
    st.dataframe(styled, use_container_width=True)

with right:
    latest_w = w_hist.iloc[-1].rename("weight").to_frame()
    st.caption(f"Latest target weights ({w_hist.index[-1].date()})")
    st.dataframe(latest_w.style.format({"weight": "{:.2%}"}), use_container_width=True)

st.caption(
    "⚠️ Research / educational tool — not investment advice. Returns are gross of "
    "taxes and slippage; transaction costs are modelled as a turnover-based drag."
)
