"""
src/eda.py  —  CRISP-DM PHASE 2 (Data Understanding: exploration)
=================================================================
Exploratory Data Analysis & data-quality diagnostics.

Provides:
  * missing-data audit,
  * trading-volume anomaly detection (robust z-score),
  * cross-asset return correlation matrix + heatmap,
  * Augmented Dickey-Fuller (ADF) stationarity tests on both raw price levels
    and log returns — the empirical justification for modeling returns rather
    than prices in Phase 3.

All plotting helpers save to ``outputs/figures`` and never call plt.show(), so
the module is safe to run headless inside the full pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # headless backend; figures are written to disk.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

import config
from src.data_ingestion import get_price_panel
from src.utils import banner, get_logger

logger = get_logger("eda")

sns.set_theme(style="whitegrid", context="notebook")


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------
def missing_data_report(long: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(ticker, field) count and percentage of missing values.

    Returns a tidy summary; also logs any field exceeding a 1% gap, which is a
    red flag for a delisted/illiquid name or a broken download.
    """
    grouped = long.isna().groupby(level="Ticker")
    counts = grouped.sum()
    totals = long.groupby(level="Ticker").size()
    pct = counts.div(totals, axis=0) * 100.0

    report = pd.concat({"missing_count": counts, "missing_pct": pct}, axis=1)
    flagged = pct[(pct > 1.0).any(axis=1)]
    if not flagged.empty:
        logger.warning("Tickers with >1%% missing in some field:\n%s", flagged)
    else:
        logger.info("Missing-data audit clean: no field exceeds 1%% gaps.")
    return report


def detect_volume_anomalies(
    long: pd.DataFrame, z_threshold: float = 5.0
) -> pd.DataFrame:
    """
    Flag trading days whose volume is a robust-z-score outlier per ticker.

    We use the *median absolute deviation* (MAD) rather than the standard
    deviation because volume is heavily right-skewed and fat-tailed; a classic
    mean/std z-score would be dragged around by the very spikes we want to
    detect. The robust z-score is:

        z_i = 0.6745 * (x_i - median(x)) / MAD(x)

    where MAD = median(|x - median(x)|) and 0.6745 rescales MAD to be a
    consistent estimator of σ for normally-distributed data.
    """
    out: list[pd.DataFrame] = []
    for ticker, frame in long.groupby(level="Ticker"):
        vol = frame["Volume"].astype(float)
        median = vol.median()
        mad = (vol - median).abs().median()
        if mad == 0:
            continue
        robust_z = 0.6745 * (vol - median) / mad
        mask = robust_z.abs() > z_threshold
        if mask.any():
            anomalies = pd.DataFrame(
                {
                    "Ticker": ticker,
                    "Volume": vol[mask].values,
                    "robust_z": robust_z[mask].values,
                },
                index=vol[mask].index.get_level_values("Date"),
            )
            out.append(anomalies)

    if not out:
        logger.info("No volume anomalies beyond |z|>%.1f.", z_threshold)
        return pd.DataFrame(columns=["Ticker", "Volume", "robust_z"])

    result = pd.concat(out).sort_index()
    logger.info("Detected %d volume anomalies (|z|>%.1f).", len(result), z_threshold)
    return result


# ---------------------------------------------------------------------------
# Correlation structure
# ---------------------------------------------------------------------------
def return_correlation(long: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of daily log returns across all symbols."""
    panel = get_price_panel(long, "Adj Close")
    log_ret = np.log(panel / panel.shift(1)).dropna(how="all")
    corr = log_ret.corr()
    return corr


def plot_correlation_heatmap(
    corr: pd.DataFrame, filename: str = "correlation_heatmap.png"
) -> str:
    """Render and save a correlation heatmap. Returns the saved path."""
    config.ensure_dirs()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Daily Log-Return Correlation")
    fig.tight_layout()
    path = config.FIGURE_DIR / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Saved correlation heatmap → %s", path)
    return str(path)


def plot_price_history(
    long: pd.DataFrame, filename: str = "price_history.png"
) -> str:
    """Plot normalised (base=100) adjusted-close trajectories for all symbols."""
    config.ensure_dirs()
    panel = get_price_panel(long, "Adj Close").dropna()
    normalised = panel / panel.iloc[0] * 100.0

    fig, ax = plt.subplots(figsize=(11, 6))
    normalised.plot(ax=ax, linewidth=1.4)
    ax.set_title("Normalised Price History (base = 100)")
    ax.set_ylabel("Index level")
    ax.set_xlabel("Date")
    ax.legend(title="Ticker", ncol=len(panel.columns))
    fig.tight_layout()
    path = config.FIGURE_DIR / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Saved price history → %s", path)
    return str(path)


# ---------------------------------------------------------------------------
# Stationarity (ADF)
# ---------------------------------------------------------------------------
@dataclass
class ADFResult:
    series: str
    statistic: float
    p_value: float
    used_lag: int
    n_obs: int
    crit_1pct: float
    crit_5pct: float
    is_stationary: bool  # at the 5% level


def adf_test(series: pd.Series, name: str, signif: float = 0.05) -> ADFResult:
    """
    Run the Augmented Dickey-Fuller test on a single series.

    Null hypothesis H0: the series has a unit root (is non-stationary).
    A p-value below `signif` lets us reject H0 → the series is stationary.

    Price levels almost always FAIL to reject (non-stationary, trending),
    whereas log returns almost always reject (stationary) — which is the whole
    reason Phase 3 models returns.
    """
    clean = series.dropna()
    stat, p_value, used_lag, n_obs, crit, _ = adfuller(clean, autolag="AIC")
    return ADFResult(
        series=name,
        statistic=float(stat),
        p_value=float(p_value),
        used_lag=int(used_lag),
        n_obs=int(n_obs),
        crit_1pct=float(crit["1%"]),
        crit_5pct=float(crit["5%"]),
        is_stationary=bool(p_value < signif),
    )


def stationarity_report(long: pd.DataFrame) -> pd.DataFrame:
    """
    ADF test on the price level AND the log return of every ticker.

    Returns a tidy DataFrame; the contrast between the two blocks is the
    empirical evidence for the return transform used downstream.
    """
    panel = get_price_panel(long, "Adj Close")
    rows: list[ADFResult] = []

    for col in panel.columns:
        price = panel[col].dropna()
        log_ret = np.log(price / price.shift(1)).dropna()
        rows.append(adf_test(price, f"{col} (price level)"))
        rows.append(adf_test(log_ret, f"{col} (log return)"))

    report = pd.DataFrame([r.__dict__ for r in rows]).set_index("series")
    n_stationary = int(report["is_stationary"].sum())
    logger.info(
        "ADF complete: %d/%d series stationary at 5%%.", n_stationary, len(report)
    )
    return report


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_eda(long: pd.DataFrame) -> dict[str, object]:
    """
    Execute the full Phase-2 EDA suite and return a dict of artifacts.

    The console output is intentionally readable as a standalone report.
    """
    logger.info(banner("PHASE 2 — DATA UNDERSTANDING / EDA"))

    missing = missing_data_report(long)
    anomalies = detect_volume_anomalies(long)
    corr = return_correlation(long)
    adf = stationarity_report(long)

    heatmap_path = plot_correlation_heatmap(corr)
    price_path = plot_price_history(long)

    print(banner("Missing-data summary"))
    print(missing.round(3).to_string())
    print(banner("Return correlation matrix"))
    print(corr.round(3).to_string())
    print(banner("Augmented Dickey-Fuller stationarity"))
    print(
        adf[["statistic", "p_value", "is_stationary"]].round(4).to_string()
    )

    return {
        "missing": missing,
        "volume_anomalies": anomalies,
        "correlation": corr,
        "adf": adf,
        "figures": {"correlation": heatmap_path, "price_history": price_path},
    }
