# 📈 Stock Market Analysis & Investment Optimization

A **production-grade, CRISP-DM structured** data-science pipeline that ingests
historical equity data, engineers a rich technical/statistical feature set,
trains next-day return & direction models, evaluates them rigorously, and
constructs **risk-optimised portfolios** via Modern Portfolio Theory —
benchmarked end-to-end against the S&P 500.

> **Disclaimer:** This is a quantitative research / decision-support tool for
> educational purposes. It is **not** investment advice.

---

## 🎯 What it does

| CRISP-DM Phase | Module(s) | Output |
|----------------|-----------|--------|
| **1. Business Understanding** | [`docs/phase1_business_understanding.md`](docs/phase1_business_understanding.md) | Objectives, success metrics, benchmarks |
| **2. Data Understanding** | `src/data_ingestion.py`, `src/eda.py` | Cached OHLCV, missing-data audit, volume anomalies, correlation heatmap, **ADF stationarity tests** |
| **3. Data Preparation** | `src/feature_engineering.py`, `src/data_splitting.py` | Leakage-free feature matrix (technical + statistical + lagged), chronological splits, **Walk-Forward Validation** |
| **4. Modeling** | `src/modeling.py` | Tuned **XGBoost/Ridge regressor** + **XGBoost/Logistic classifier**, **heavily regularised** (L1/L2/gamma + subsampling), **walk-forward CV** |
| **5. Evaluation** | `src/evaluation.py`, `src/portfolio_optimization.py`, `src/performance.py` | Confusion matrices, ROC-AUC, MAE/RMSE, **SHAP explainability**, **efficient frontier**, Max-Sharpe & Min-Variance weights, **dynamic rolling rebalancing backtest** with transaction costs, Sharpe/Sortino/Drawdown vs SPY |
| **6. Deployment** | `main.py`, **`app.py`** | One-command end-to-end pipeline + **interactive Streamlit dashboard** |

### 🏛️ Institutional-grade enhancements

| # | Enhancement | Where |
|---|-------------|-------|
| **1** | **Walk-Forward Validation** (rolling 12m-train / 1m-test) replaces static CV; **heavy XGBoost regularisation** (shallow `max_depth∈{1,2,3}`, L1 `reg_alpha`, L2 `reg_lambda`, split-penalty `gamma>0`, 0.7 row/column subsampling) | `data_splitting.py`, `modeling.py` |
| **2** | **Dynamic rolling rebalancing engine** — monthly/quarterly re-optimisation on trailing windows, drifting weights, **transaction-cost drag (5–10 bps on turnover)**, compounding net equity curve | `portfolio_optimization.py`, `performance.py` |
| **3** | **SHAP explainability** (TreeExplainer) + native gain/weight/cover — ranks which indicators carry signal vs noise; exports plots + CSVs | `evaluation.py` |
| **4** | **Streamlit dashboard** — interactive Plotly equity curve vs SPY, rolling allocations, efficient frontier, live scorecards | `app.py` |
| **5** | **Signal-driven allocation** — walk-forward (periodic-refit) out-of-sample model predictions feed the optimiser as a time-varying μ (`signal_mu_provider`) or tilt a base allocation (`signal_tilt_transform`); backtested vs pure-MPT vs SPY. Strictly no look-ahead (proven by test). | `signals.py` |
| **6** | **Robustness layer** — Ledoit-Wolf covariance shrinkage; risk overlays (per-name position cap via water-filling, volatility targeting with a cash leg, drawdown stop); and honest walk-forward (rolling-retrain) OOS model metrics. All overlays opt-in; OFF reduces exactly to baseline (tested). | `portfolio_optimization.py`, `evaluation.py` |

---

## 🧱 Architecture

```
Stock Market Analysis/
├── main.py                       # Phase 6 — end-to-end orchestrator (CLI)
├── app.py                        # Phase 6 — interactive Streamlit dashboard
├── config.py                     # Single source of truth: tickers, windows, params
├── requirements.txt
├── README.md
├── docs/
│   └── phase1_business_understanding.md
├── src/
│   ├── utils.py                  # logging, timing, banners
│   ├── data_ingestion.py         # Phase 2 — yfinance pull → tidy parquet
│   ├── eda.py                    # Phase 2 — EDA + ADF stationarity
│   ├── feature_engineering.py    # Phase 3 — indicators from first principles
│   ├── data_splitting.py         # Phase 3 — chronological / TimeSeriesSplit
│   ├── modeling.py               # Phase 4 — dual model + CV tuning
│   ├── signals.py                # Bridge — walk-forward signal-driven allocation
│   ├── evaluation.py             # Phase 5 — predictive metrics & plots
│   ├── portfolio_optimization.py # Phase 5 — MPT engine (SciPy)
│   └── performance.py            # Phase 5 — Sharpe/Sortino/Drawdown vs SPY
├── data/{raw,processed}/         # cached pulls & feature matrices
└── outputs/{figures,models,reports}/
```

Every module corresponds to exactly one CRISP-DM concern and imports its
configuration from `config.py`, so a single edit re-wires the whole run.

---

## 🚀 Quickstart

```bash
# 1. (recommended) create an isolated environment
python -m venv .venv
.\.venv\Scripts\activate        # Windows PowerShell
# source .venv/bin/activate     # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. run the full pipeline (downloads ~8y of daily data on first run)
python main.py
```

First run downloads data from Yahoo Finance and caches it to
`data/raw/ohlcv_long.parquet`; subsequent runs are offline unless you pass
`--refresh`.

### Common invocations

```bash
python main.py                          # full pipeline, default universe
python main.py --tickers AAPL MSFT NVDA # custom universe (SPY added automatically)
python main.py --start 2019-01-01       # custom window
python main.py --skip-models            # data + EDA + optimization only (fast)
python main.py --model-implied-mu       # feed model-predicted returns into the optimizer
python main.py --rebalance-freq Q       # quarterly (vs default monthly) rebalancing
python main.py --no-shap                # native gain importances instead of SHAP
python main.py --signal-driven          # add model-signal strategies to the backtest
python main.py --signal-driven --signal-engine xgboost   # (slower) tree-based signal
python main.py --shrinkage              # Ledoit-Wolf covariance in the backtest
python main.py --target-vol 0.15        # volatility-targeting overlay (15% annual)
python main.py --max-weight 0.35        # cap any single position at 35%
python main.py --drawdown-stop 0.25     # de-risk to cash on a 25% drawdown
python main.py --wf-eval                # honest walk-forward (rolling-retrain) metrics
python main.py --refresh --verbose      # re-download + DEBUG logging
```

### Interactive dashboard

```bash
streamlit run app.py
```

Pick the universe, rebalancing frequency, estimation lookback, transaction
cost, and strategy in the sidebar, then click **Run analysis** to render the
dynamic equity curve vs SPY, rolling allocations, the efficient frontier, and
the risk-adjusted scorecard — all interactive Plotly.

---

## 🔬 Methodology highlights

### Leakage prevention (taken seriously)
- **Returns, not prices** — modeled after ADF tests confirm price levels are
  non-stationary while log returns are stationary.
- **Lagged features only** for past information; the prediction target is the
  *only* forward-looking column and is created with an explicit `shift(-1)`.
- **Chronological splits** — the most recent 20% is an untouched hold-out test
  set; hyper-parameters are tuned with a **rolling Walk-Forward Validator**
  (fixed 12-month train / 1-month test window sliding forward one month at a
  time — every validation fold lies strictly after its training fold, exactly
  mirroring periodic production re-fits).
- **Rolling portfolio backtest** — at each monthly/quarterly rebalance the
  optimiser sees only the trailing estimation window (`.loc[:date]`), so the
  out-of-sample equity curve never peeks at the future.

### Feature catalogue (Phase 3)
- **Technical:** SMA/EMA (20/50/200, as price ratios), RSI(14, Wilder),
  MACD(12/26/9), Bollinger %b & width, ATR(14, %-of-price).
- **Statistical:** annualised rolling volatility, rolling skewness, rolling
  CAPM beta vs SPY.
- **Lagged:** log returns & log-volume changes at t-1, t-2, t-5.

### Portfolio optimization (Phase 5)
Pure SciPy `SLSQP` implementation of Markowitz mean-variance optimisation —
no black-box dependency. Solves for the **Maximum Sharpe (tangency)** and
**Global Minimum Variance** portfolios and traces the full **efficient
frontier** by minimising variance across a grid of target returns.

```
portfolio return    μ_p = wᵀμ
portfolio variance  σ²_p = wᵀΣw
Sharpe              S   = (μ_p − r_f) / σ_p
```

---

## 📊 Outputs

After a run, inspect:

- `outputs/figures/` — price history, correlation heatmap, per-ticker confusion
  matrices & ROC curves, **efficient_frontier.png**
- `outputs/reports/` — `model_evaluation_summary.csv`, `portfolio_weights.csv`,
  `efficient_frontier.csv`, `portfolio_performance.csv`, `pipeline.log`
- `outputs/models/` — serialized tuned estimators (`*.joblib`)

---

## ⚙️ Configuration

All knobs live in `config.py`: the investable universe, date window,
risk-free rate, every feature window, model engine (`"xgboost"` vs `"linear"`),
CV fold count, and portfolio constraints (long-only vs shorting, weight
bounds). Change once, re-run, fully reproducible.

---

## 🧩 Tech stack

`Python 3.11+` · `yfinance` · `pandas` · `numpy` · `scipy` · `scikit-learn` ·
`xgboost` · `statsmodels` · `matplotlib` · `seaborn`

---

## 🛣️ Extension hooks

- Swap historical μ for model-implied expected returns (`--model-implied-mu`).
- Add transaction costs / turnover penalties in `src/performance.py`.
- Add a Ledoit-Wolf shrinkage estimator for Σ in `portfolio_optimization.py`.
- Add walk-forward backtesting of the optimised weights over rolling windows.
- Wrap `main.py` in a scheduler / Streamlit dashboard for live monitoring.
