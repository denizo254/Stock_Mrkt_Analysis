"""
src/data_ingestion.py  —  CRISP-DM PHASE 2 (Data Understanding: acquisition)
============================================================================
Robust ingestion of historical daily OHLCV data via ``yfinance``.

Design goals
------------
* **Deterministic on disk.** Raw pulls are cached to parquet so downstream
  phases never re-hit the network unless explicitly asked to refresh.
* **Tidy long format.** We persist a single long DataFrame indexed by
  (Date, Ticker) with columns [Open, High, Low, Close, Adj Close, Volume].
  This is trivially filterable per-ticker and avoids the brittle MultiIndex
  column layout yfinance returns for multi-symbol downloads.
* **Defensive.** Missing symbols, empty pulls, and partial failures are logged
  loudly rather than silently producing a half-empty dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

import config
from src.utils import get_logger, timed

logger = get_logger("data_ingestion")

_RAW_PARQUET = config.RAW_DIR / "ohlcv_long.parquet"

# Canonical column order used everywhere downstream.
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _flatten_yf_download(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """
    Convert yfinance's wide MultiIndex frame into a tidy long frame.

    yfinance returns columns as a MultiIndex of (Field, Ticker) when several
    symbols are requested, or a flat (Field) index for a single symbol. This
    helper normalises both cases into:

        index: DatetimeIndex named 'Date'
        columns: ['Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    """
    frames: list[pd.DataFrame] = []

    if isinstance(raw.columns, pd.MultiIndex):
        # Columns look like ('Close', 'AAPL'). Level 0 = field, level 1 = ticker.
        for sym in symbols:
            # Guard against symbols that silently failed to download.
            if sym not in raw.columns.get_level_values(1):
                logger.warning("No data returned for symbol '%s' — skipping.", sym)
                continue
            sub = raw.xs(sym, axis=1, level=1).copy()
            sub["Ticker"] = sym
            frames.append(sub)
    else:
        # Single-symbol pull: flat columns, one ticker.
        sub = raw.copy()
        sub["Ticker"] = symbols[0]
        frames.append(sub)

    if not frames:
        raise RuntimeError("yfinance returned no usable data for any symbol.")

    long = pd.concat(frames, axis=0)
    long.index.name = "Date"

    # yfinance occasionally omits 'Adj Close' when auto_adjust=True; add a
    # graceful fallback so the schema is stable.
    if "Adj Close" not in long.columns:
        long["Adj Close"] = long["Close"]

    long = long.reset_index().set_index(["Date", "Ticker"]).sort_index()
    return long[OHLCV_COLUMNS]


def download_market_data(
    symbols: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    interval: str = config.DATA_INTERVAL,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download (or load cached) historical OHLCV data.

    Parameters
    ----------
    symbols : list of tickers. Defaults to ``config.ALL_SYMBOLS``.
    start, end : ISO date strings. Default to the configured window.
    interval : bar size, e.g. '1d'.
    force_refresh : when True, ignore the on-disk cache and re-download.

    Returns
    -------
    pd.DataFrame
        Long frame indexed by (Date, Ticker) with OHLCV columns.
    """
    config.ensure_dirs()
    symbols = symbols or config.ALL_SYMBOLS
    start = start or config.START_DATE
    end = end or config.END_DATE

    if _RAW_PARQUET.exists() and not force_refresh:
        logger.info("Loading cached raw data from %s", _RAW_PARQUET)
        cached = pd.read_parquet(_RAW_PARQUET)
        have = set(cached.index.get_level_values("Ticker").unique())
        if set(symbols).issubset(have):
            return cached
        logger.info(
            "Cache missing symbols %s — re-downloading.", set(symbols) - have
        )

    with timed(f"Downloading {len(symbols)} symbols from yfinance", logger):
        raw = yf.download(
            tickers=symbols,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,   # keep both Close and Adj Close explicitly
            group_by="column",
            progress=False,
            threads=True,
        )

    if raw is None or raw.empty:
        raise RuntimeError(
            "yfinance download returned no data. Check connectivity, the "
            "ticker spellings, and the date range."
        )

    long = _flatten_yf_download(raw, symbols)
    long.to_parquet(_RAW_PARQUET)
    logger.info(
        "Saved %d rows across %d symbols to %s",
        len(long),
        long.index.get_level_values("Ticker").nunique(),
        _RAW_PARQUET,
    )
    return long


def get_price_panel(
    long: pd.DataFrame, field: str = "Adj Close"
) -> pd.DataFrame:
    """
    Pivot the long frame into a wide price panel: index=Date, columns=Ticker.

    'Adj Close' is the correct field for return calculations because it folds
    in dividends and splits, giving a continuous total-return series.
    """
    if field not in long.columns:
        raise KeyError(f"Field '{field}' not present in data: {list(long.columns)}")
    panel = long[field].unstack("Ticker").sort_index()
    return panel


def get_ticker_frame(long: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return the full OHLCV frame for a single ticker, indexed by Date."""
    if ticker not in long.index.get_level_values("Ticker"):
        raise KeyError(f"Ticker '{ticker}' not in dataset.")
    frame = long.xs(ticker, level="Ticker").sort_index()
    return frame


if __name__ == "__main__":  # pragma: no cover — manual smoke test
    df = download_market_data()
    print(df.head())
    print("\nPrice panel:\n", get_price_panel(df).tail())
