"""
src/data_splitting.py  —  CRISP-DM PHASE 3 (Data Preparation: splitting)
========================================================================
Chronological, leakage-proof partitioning of the feature matrix.

Why not ``train_test_split(shuffle=True)``?
-------------------------------------------
Financial features are autocorrelated and the data-generating process drifts
over time (regime change). Shuffling would let the model "see the future":
a randomly-selected training row dated 2024 could inform a prediction about a
test row dated 2019. We therefore:

  1. Hold out the most recent ``test_size`` fraction as an untouched final
     test set (simulating live deployment on unseen future data).
  2. Tune hyper-parameters on the earlier portion using scikit-learn's
     ``TimeSeriesSplit`` — an expanding-window scheme where every validation
     fold lies strictly *after* its training fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import config
from src.utils import get_logger

logger = get_logger("data_splitting")


@dataclass
class SplitData:
    """Container for a single train/test partition of one target."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def train_span(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.X_train.index.min(), self.X_train.index.max()

    @property
    def test_span(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.X_test.index.min(), self.X_test.index.max()


def chronological_split(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_size: float = config.MODEL.test_size,
) -> SplitData:
    """
    Split a feature matrix into train/test by time order (no shuffling).

    The last ``test_size`` fraction of rows (latest dates) becomes the test
    set. This mirrors deployment: train on history, predict the future.
    """
    n = len(feature_df)
    split_idx = int(n * (1.0 - test_size))

    X = feature_df[feature_cols]
    y = feature_df[target_col]

    split = SplitData(
        X_train=X.iloc[:split_idx],
        X_test=X.iloc[split_idx:],
        y_train=y.iloc[:split_idx],
        y_test=y.iloc[split_idx:],
    )
    logger.info(
        "Split '%s': train=%d (%s → %s)  test=%d (%s → %s)",
        target_col,
        len(split.X_train),
        split.train_span[0].date(),
        split.train_span[1].date(),
        len(split.X_test),
        split.test_span[0].date(),
        split.test_span[1].date(),
    )
    return split


def make_ts_cv(n_splits: int = config.MODEL.n_splits) -> TimeSeriesSplit:
    """
    Build the expanding-window cross-validator used for hyper-parameter search.

    Each successive fold trains on all data up to a cut point and validates on
    the immediately-following block — never the reverse. This is the only
    cross-validation scheme that respects causality for time series.
    """
    return TimeSeriesSplit(n_splits=n_splits)
