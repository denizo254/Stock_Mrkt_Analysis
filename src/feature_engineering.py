"""
src/feature_engineering.py  —  CRISP-DM PHASE 3 (Data Preparation)
==================================================================
Transforms raw OHLCV bars into a leakage-free, model-ready feature matrix.

Every indicator is implemented from first principles (no TA-Lib dependency) so
the mathematics is auditable and the install footprint stays small.

The cardinal rule of this module: **a feature row dated t may only use
information observable at or before the close of day t.** The prediction
targets (next-day log return / direction) are the only forward-looking
columns, and they are produced by an explicit ``shift(-1)`` that is dropped
before any model ever sees a feature.

Indicator catalogue
--------------------
Technical : SMA(20/50/200), EMA(20/50/200), RSI(14), MACD(12,26,9),
            Bollinger Bands(20, 2σ), ATR(14)
Statistical: rolling volatility, rolling skewness, rolling beta vs SPY
Lagged     : lagged log returns & log-volume changes (t-1, t-2, t-5)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.data_ingestion import get_price_panel, get_ticker_frame
from src.utils import get_logger

logger = get_logger("feature_engineering")

FCFG = config.FEATURES


# ===========================================================================
# 1. Return transform (handles non-stationarity, per Phase-2 ADF evidence)
# ===========================================================================
def log_returns(close: pd.Series) -> pd.Series:
    """
    Daily log return:  r_t = ln(P_t / P_{t-1}).

    Log returns are time-additive and approximately stationary, which is why
    they are the modeling unit rather than raw prices.
    """
    return np.log(close / close.shift(1))


# ===========================================================================
# 2. Technical indicators
# ===========================================================================
def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over `window` bars."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (span convention, α = 2/(span+1))."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing.

        RS  = avg_gain / avg_loss   (Wilder EMA, α = 1/period)
        RSI = 100 - 100 / (1 + RS)

    Bounded in [0, 100]; >70 conventionally "overbought", <30 "oversold".
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing == EWM with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0 (pure uptrend) RSI saturates at 100.
    out = out.where(avg_loss != 0.0, 100.0)
    return out


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence.

        MACD line   = EMA_fast(close) - EMA_slow(close)
        Signal line = EMA_signal(MACD line)
        Histogram   = MACD line - Signal line
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist}
    )


def bollinger_bands(
    close: pd.Series, window: int = 20, k: float = 2.0
) -> pd.DataFrame:
    """
    Bollinger Bands around an SMA mid-line.

        mid   = SMA(close, window)
        upper = mid + k * σ
        lower = mid - k * σ
        %b    = (close - lower) / (upper - lower)   ∈ ~[0, 1]
        width = (upper - lower) / mid               (volatility proxy)

    %b and width are the *stationary, scale-free* features we feed the model
    (the raw bands themselves are price-scaled and would leak the level).
    """
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    pct_b = (close - lower) / (upper - lower)
    width = (upper - lower) / mid
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_pct_b": pct_b,
            "bb_width": width,
        }
    )


def average_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Average True Range — a directional-agnostic volatility measure.

        TR_t = max( High_t - Low_t,
                    |High_t - Close_{t-1}|,
                    |Low_t  - Close_{t-1}| )
        ATR  = Wilder EMA of TR over `period`.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr


# ===========================================================================
# 3. Statistical features
# ===========================================================================
def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Annualised rolling standard deviation of returns."""
    return returns.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(
        config.TRADING_DAYS_PER_YEAR
    )


def rolling_skewness(returns: pd.Series, window: int) -> pd.Series:
    """Rolling sample skewness of returns (asymmetry of the return distn)."""
    return returns.rolling(window, min_periods=window).skew()


def rolling_beta(
    asset_returns: pd.Series, market_returns: pd.Series, window: int
) -> pd.Series:
    """
    Rolling CAPM beta of the asset against the market benchmark.

        β_t = Cov(r_asset, r_mkt)_t / Var(r_mkt)_t      over a trailing window.

    Beta > 1 ⇒ more volatile than the market; < 1 ⇒ defensive.
    """
    aligned = pd.concat([asset_returns, market_returns], axis=1).dropna()
    aligned.columns = ["asset", "market"]
    cov = aligned["asset"].rolling(window, min_periods=window).cov(aligned["market"])
    var = aligned["market"].rolling(window, min_periods=window).var(ddof=0)
    beta = cov / var
    return beta.reindex(asset_returns.index)


# ===========================================================================
# 4. Lagged features (explicit anti-leakage)
# ===========================================================================
def add_lagged_features(
    df: pd.DataFrame, columns: list[str], lags: tuple[int, ...]
) -> pd.DataFrame:
    """
    Append shifted copies of `columns` for each lag in `lags`.

    A lag-k feature at row t holds the value from row t-k, i.e. strictly past
    information — this is what makes the design matrix causally valid.
    """
    for col in columns:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


# ===========================================================================
# 5. Assemble the per-ticker feature matrix
# ===========================================================================
def build_features_for_ticker(
    long: pd.DataFrame,
    ticker: str,
    benchmark: str = config.BENCHMARK,
) -> pd.DataFrame:
    """
    Build the complete feature + target matrix for one ticker.

    Returns a DataFrame indexed by Date whose columns are the engineered
    features plus two targets:
        * ``target_logret``  — next-day log return  (regression label)
        * ``target_dir``     — 1 if next-day return > 0 else 0 (classification)

    All rows with NaNs introduced by warm-up windows or the target shift are
    dropped, leaving a dense, leakage-free matrix.
    """
    frame = get_ticker_frame(long, ticker)
    bench = get_ticker_frame(long, benchmark)

    close = frame["Adj Close"]
    high, low = frame["High"], frame["Low"]
    volume = frame["Volume"].astype(float)

    feat = pd.DataFrame(index=frame.index)

    # --- core return series -------------------------------------------------
    feat["log_return"] = log_returns(close)
    feat["log_volume_chg"] = np.log(volume / volume.shift(1)).replace(
        [np.inf, -np.inf], np.nan
    )

    # --- trend / moving averages -------------------------------------------
    for w in FCFG.sma_windows:
        # Ratio of price to its SMA is scale-free and stationary-ish.
        feat[f"price_sma{w}_ratio"] = close / sma(close, w) - 1.0
    for w in FCFG.ema_windows:
        feat[f"price_ema{w}_ratio"] = close / ema(close, w) - 1.0

    # --- momentum / oscillators --------------------------------------------
    feat["rsi"] = rsi(close, FCFG.rsi_period)
    feat = feat.join(macd(close, FCFG.macd_fast, FCFG.macd_slow, FCFG.macd_signal))

    # --- volatility / bands -------------------------------------------------
    bb = bollinger_bands(close, FCFG.bollinger_window, FCFG.bollinger_k)
    feat["bb_pct_b"] = bb["bb_pct_b"]
    feat["bb_width"] = bb["bb_width"]
    feat["atr"] = average_true_range(high, low, close, FCFG.atr_period)
    # Normalise ATR by price so it is comparable across tickers / regimes.
    feat["atr_pct"] = feat["atr"] / close

    # --- statistical --------------------------------------------------------
    feat["volatility"] = rolling_volatility(feat["log_return"], FCFG.volatility_window)
    feat["skewness"] = rolling_skewness(feat["log_return"], FCFG.skew_window)
    bench_ret = log_returns(bench["Adj Close"])
    feat["beta"] = rolling_beta(feat["log_return"], bench_ret, FCFG.beta_window)

    # --- lagged (strictly past) --------------------------------------------
    feat = add_lagged_features(
        feat, ["log_return"], FCFG.return_lags
    )
    feat = add_lagged_features(
        feat, ["log_volume_chg"], FCFG.volume_lags
    )

    # --- targets (the ONLY forward-looking columns) ------------------------
    feat["target_logret"] = feat["log_return"].shift(-1)
    feat["target_dir"] = (feat["target_logret"] > 0).astype("int8")

    before = len(feat)
    feat = feat.dropna()
    logger.info(
        "[%s] feature matrix: %d rows (dropped %d warm-up/edge rows), %d features",
        ticker,
        len(feat),
        before - len(feat),
        feat.shape[1] - 2,  # minus the two targets
    )
    return feat


def feature_columns(feature_df: pd.DataFrame) -> list[str]:
    """Return the model-input columns (everything that is not a target)."""
    return [c for c in feature_df.columns if not c.startswith("target_")]


def build_all_features(
    long: pd.DataFrame, tickers: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """
    Build feature matrices for every ticker in the investable universe and
    persist each to ``data/processed/<ticker>_features.parquet``.
    """
    config.ensure_dirs()
    tickers = tickers or config.TICKERS
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        fdf = build_features_for_ticker(long, t)
        fdf.to_parquet(config.PROCESSED_DIR / f"{t}_features.parquet")
        out[t] = fdf
    return out
