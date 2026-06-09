"""
main.py  —  CRISP-DM PHASE 6 (Deployment & Wrapper)
===================================================
End-to-end orchestration of the full data-to-optimization pipeline:

    Phase 2  Ingest market data (yfinance)            ──► data/raw
    Phase 2  Exploratory analysis + ADF stationarity  ──► outputs/figures
    Phase 3  Feature engineering (per ticker)         ──► data/processed
    Phase 4  Train regression + classification models ──► outputs/models
    Phase 5  Evaluate models (metrics, ROC, CM)       ──► outputs/reports
    Phase 5  Mean-variance portfolio optimization     ──► outputs/reports
    Phase 5  Risk-adjusted performance vs SPY         ──► outputs/reports

Usage
-----
    python main.py                       # full pipeline, default universe
    python main.py --tickers AAPL MSFT   # custom universe
    python main.py --skip-models         # data + EDA + optimization only
    python main.py --refresh             # force re-download from yfinance
    python main.py --model-implied-mu    # feed model returns into the optimizer

Every stage logs to stdout and to outputs/reports/pipeline.log.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging

import pandas as pd

import config
from src.data_ingestion import download_market_data
from src.data_splitting import chronological_split
from src.eda import run_eda
from src.evaluation import (
    evaluate_classification,
    evaluate_regression,
    evaluation_summary,
    explain_model,
    plot_confusion_matrix,
    plot_roc_curve,
)
from src.feature_engineering import build_features_for_ticker, feature_columns
from src.modeling import predict_regression, train_classifier, train_regressor
from src.performance import (
    benchmark_report,
    dynamic_benchmark_report,
    plot_dynamic_equity_curves,
)
from src.portfolio_optimization import optimize_portfolio, rolling_rebalance_backtest
from src.signals import run_signal_strategies
from src.utils import banner, configure_logging, get_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stock Market Analysis & Investment Optimization pipeline."
    )
    p.add_argument("--tickers", nargs="+", default=config.TICKERS,
                   help="Investable universe (benchmark SPY is added automatically).")
    p.add_argument("--start", default=config.START_DATE, help="Start date YYYY-MM-DD.")
    p.add_argument("--end", default=config.END_DATE, help="End date YYYY-MM-DD.")
    p.add_argument("--refresh", action="store_true",
                   help="Force re-download even if a cache exists.")
    p.add_argument("--skip-eda", action="store_true", help="Skip Phase 2 EDA.")
    p.add_argument("--skip-models", action="store_true",
                   help="Skip Phase 4 modeling + Phase 5 model evaluation.")
    p.add_argument("--model-implied-mu", action="store_true",
                   help="Use model-predicted mean returns as μ in the optimizer.")
    p.add_argument("--rebalance-freq", choices=["M", "Q"], default=config.REBALANCE.frequency,
                   help="Rolling rebalancing frequency: M=monthly, Q=quarterly.")
    p.add_argument("--no-shap", action="store_true",
                   help="Disable SHAP explainability (use native gain only).")
    p.add_argument("--signal-driven", action="store_true",
                   help="Add walk-forward signal-driven strategies (model μ + tilt) "
                        "to the dynamic backtest comparison.")
    p.add_argument("--signal-engine", choices=["linear", "xgboost"], default="linear",
                   help="Model engine for the walk-forward signal (default: linear, fast).")
    p.add_argument("--verbose", action="store_true", help="DEBUG-level logging.")
    return p.parse_args()


def run_modeling_stage(
    long: pd.DataFrame, tickers: list[str], logger: logging.Logger
) -> dict[str, float]:
    """
    Phase 4 + Phase 5(a): train and evaluate both models for each ticker.

    Returns a mapping ticker → annualised model-implied expected return, built
    from the mean of the regression model's predictions on the test set. This
    can optionally seed the portfolio optimizer's μ vector.
    """
    logger.info(banner("PHASE 4 — MODELING  /  PHASE 5 — MODEL EVALUATION"))
    reg_reports, clf_reports = [], []
    model_implied_mu: dict[str, float] = {}

    for ticker in tickers:
        features = build_features_for_ticker(long, ticker)
        cols = feature_columns(features)

        # --- Regression: next-day log return -------------------------------
        reg_split = chronological_split(features, cols, "target_logret")
        reg_model = train_regressor(reg_split, ticker)
        reg_model.save()
        reg_reports.append(
            evaluate_regression(reg_model, reg_split.X_test, reg_split.y_test)
        )

        # Model-implied annualised expected return (mean predicted daily logret).
        preds = predict_regression(reg_model, reg_split.X_test)
        model_implied_mu[ticker] = float(preds.mean()) * config.TRADING_DAYS_PER_YEAR

        # --- Classification: next-day direction -----------------------------
        clf_split = chronological_split(features, cols, "target_dir")
        clf_model = train_classifier(clf_split, ticker)
        clf_model.save()
        clf_rep = evaluate_classification(clf_model, clf_split.X_test, clf_split.y_test)
        clf_reports.append(clf_rep)

        # Diagnostic plots.
        plot_confusion_matrix(clf_rep)
        plot_roc_curve(clf_model, clf_split.X_test, clf_split.y_test)

        # Step 3 — explainability: which indicators carry the signal?
        explain_model(reg_model, reg_split.X_test)
        explain_model(clf_model, clf_split.X_test)

    evaluation_summary(reg_reports, clf_reports)
    return model_implied_mu


def main() -> None:
    args = parse_args()
    config.ensure_dirs()
    configure_logging(
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=config.REPORT_DIR / "pipeline.log",
    )
    logger = get_logger("main")

    # Honour --no-shap by swapping in a tweaked (still-frozen) EXPLAIN config.
    if args.no_shap:
        config.EXPLAIN = dataclasses.replace(config.EXPLAIN, enable_shap=False)

    logger.info(banner("STOCK MARKET ANALYSIS & INVESTMENT OPTIMIZATION", char="#"))
    logger.info("Universe: %s | Benchmark: %s", args.tickers, config.BENCHMARK)

    # ---- Phase 2: ingest --------------------------------------------------
    symbols = list(dict.fromkeys(args.tickers + [config.BENCHMARK]))
    long = download_market_data(
        symbols=symbols, start=args.start, end=args.end, force_refresh=args.refresh
    )

    # ---- Phase 2: EDA -----------------------------------------------------
    if not args.skip_eda:
        run_eda(long)

    # ---- Phase 4 + 5(a): modeling + model evaluation ----------------------
    model_implied_mu: dict[str, float] = {}
    if not args.skip_models:
        model_implied_mu = run_modeling_stage(long, args.tickers, logger)

    # ---- Phase 5(b): portfolio optimization -------------------------------
    mu_override = None
    if args.model_implied_mu and model_implied_mu:
        mu_override = pd.Series(model_implied_mu).reindex(args.tickers)
        logger.info("Using model-implied expected returns for optimization:\n%s",
                    mu_override.round(4).to_string())

    # Static (full-sample) snapshot — efficient frontier + reference weights.
    opt = optimize_portfolio(
        long, tickers=args.tickers, expected_returns_override=mu_override
    )

    # ---- Phase 5(c): STATIC portfolio performance vs benchmark ------------
    benchmark_report(
        long,
        {
            "Max Sharpe": opt.max_sharpe.weights,
            "Min Variance": opt.min_variance.weights,
        },
    )

    # ---- Phase 5(d): DYNAMIC rolling rebalancing backtest (Step 2) --------
    logger.info(banner("PHASE 5 — DYNAMIC ROLLING REBALANCING BACKTEST"))
    backtests = {
        "Max Sharpe": rolling_rebalance_backtest(
            long, tickers=args.tickers, strategy="max_sharpe",
            frequency=args.rebalance_freq, mu_override=mu_override,
        ),
        "Min Variance": rolling_rebalance_backtest(
            long, tickers=args.tickers, strategy="min_variance",
            frequency=args.rebalance_freq,
        ),
    }

    # ---- Phase 5(e): SIGNAL-DRIVEN allocation (models -> optimizer) -------
    if args.signal_driven:
        signal_bts, _ = run_signal_strategies(
            long, tickers=args.tickers, frequency=args.rebalance_freq,
            engine=args.signal_engine,
        )
        backtests.update(signal_bts)

    dynamic_benchmark_report(long, backtests)
    plot_dynamic_equity_curves(long, backtests)

    logger.info(banner("PIPELINE COMPLETE", char="#"))
    logger.info("Artifacts written under: %s", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
