# Phase 1 — Business Understanding

> CRISP-DM mandates that we fix *what success means* before touching data. This
> document is the contract the rest of the pipeline is held to.

---

## 1. Problem framing (the investor's standpoint)

A capital allocator with a fixed budget must decide **(a)** which assets to
hold and **(b)** in what proportion, under two competing pressures:

| Pressure | Investor concern | Quantitative proxy |
|----------|------------------|--------------------|
| **Maximise return** | "Grow my capital." | Annualised return, Sharpe ratio |
| **Minimise risk** | "Don't blow up / let me sleep at night." | Volatility, **Maximum Drawdown**, Sortino |

These goals trade off against each other. Modern Portfolio Theory formalises
the trade-off via the **efficient frontier**: the set of portfolios offering
the maximum expected return for each level of risk. Our job is to (1) forecast
the short-term return signal and (2) construct a portfolio that sits on — or
near — that frontier.

### Primary objectives

1. **Predictive signal.** Forecast next-day return / direction for each asset
   with enough skill to be *directionally useful* (better than a coin flip).
2. **Risk-aware allocation.** Produce concrete weight allocations that
   **maximise the Sharpe ratio** (best risk-adjusted return) and, as a
   defensive alternative, **minimise variance** (lowest possible risk).
3. **Beat the benchmark.** Deliver a superior **risk-adjusted** profile versus
   simply buying the S&P 500 (SPY) — measured by Sharpe, Sortino, and a
   shallower maximum drawdown.

### Explicit non-goals (scope guardrails)

* No intraday / high-frequency trading — daily bars only.
* No transaction-cost, slippage, or tax modeling in the base pipeline (the
  performance layer is a gross-return approximation; hooks are noted for
  extension).
* No leverage or derivatives — long-only, fully-invested weights by default.
* This is **decision support, not investment advice.**

---

## 2. Success metrics & benchmarks

Two scorecards, because a good predictor and a good portfolio are different
things.

### A. Predictive performance (Phase 4 / 5)

| Metric | Task | Definition | Target |
|--------|------|------------|--------|
| **Directional Accuracy** | both | % of days `sign(pred) == sign(actual)` | **> 52%** (edge over 50% coin flip) |
| **MAE** | regression | mean \|pred − actual\| of daily log return | as low as possible; report vs naïve |
| **RMSE** | regression | √mean((pred − actual)²) | as low as possible |
| **ROC-AUC** | classification | area under ROC curve | **> 0.55** |
| **UP-day Precision** | classification | when model says UP, P(actually UP) | **> 0.55** |

> **Reality check.** Daily equity returns are *near* random walks; R² close to
> zero is expected and not a failure. A small, *stable* directional edge
> (52–55%) is economically meaningful when compounded and risk-managed. We
> explicitly benchmark against the naïve "predict zero / predict yesterday's
> sign" baselines.

### B. Portfolio performance (Phase 5), benchmarked vs SPY

| Metric | Definition | Success criterion |
|--------|------------|-------------------|
| **Sharpe ratio** | (μ − r_f) / σ | **> SPY's Sharpe** |
| **Sortino ratio** | (μ − r_f) / σ_downside | **> SPY's Sortino** |
| **Maximum Drawdown** | worst peak-to-trough equity decline | **shallower (less negative) than SPY**, especially for the Min-Variance portfolio |
| **Calmar ratio** | annualised return / \|max drawdown\| | **> SPY's Calmar** |

Risk-free rate assumption: **`RISK_FREE_RATE = 4.0%`** annual (configurable in
`config.py`).

---

## 3. Why these models & methods

* **Dual model (regression + classification).** Regression gives a *magnitude*
  signal that can seed expected returns; classification gives a cleaner
  *direction* signal for entry decisions. Reporting both guards against
  over-reading either.
* **XGBoost / linear fallbacks.** Gradient-boosted trees capture non-linear
  feature interactions common in technical signals; Ridge / Logistic provide
  an interpretable, low-variance baseline (and a dependency-free fallback).
* **Mean-Variance Optimization.** The canonical, transparent framework for the
  risk/return trade-off, with closed-form-adjacent SciPy optimisation so every
  constraint is auditable.

---

## 4. Key risks & assumptions

| Risk | Mitigation in the pipeline |
|------|----------------------------|
| **Look-ahead / data leakage** | Lagged features only; chronological splits; `TimeSeriesSplit` CV; targets created by explicit `shift(-1)` then dropped. |
| **Non-stationarity** | Model log returns (validated by ADF in Phase 2), use scale-free ratio features. |
| **Overfitting to one regime** | 8-year window spanning multiple regimes; expanding-window CV; held-out most-recent test set. |
| **Estimation error in μ, Σ** | μ is the noisiest input — Min-Variance (μ-free) offered as a robust alternative to Max-Sharpe. |
| **Survivorship / point-in-time** | Documented limitation; current universe is large-cap survivors. |

---

## 5. Definition of done

The project is "successful" for this iteration when the pipeline runs
end-to-end (`python main.py`) and produces: stationarity evidence, per-ticker
model scorecards meeting the predictive targets above on the held-out test set,
an efficient-frontier plot, explicit Max-Sharpe & Min-Variance weight vectors,
and a performance scorecard demonstrating the risk-adjusted comparison against
SPY.
