"""
src/data_splitting.py  —  CRISP-DM PHASE 3 (Data Preparation: splitting)
========================================================================
Chronological, leakage-proof partitioning + **Walk-Forward Validation (WFV)**.

Two responsibilities:

  1. ``chronological_split`` — carve off the most-recent ``test_size`` fraction
     as an untouched final hold-out (simulates live deployment on the future).

  2. ``WalkForwardValidator`` / ``make_walk_forward_cv`` — the rolling
     cross-validator used for hyper-parameter tuning. This REPLACES the old
     static ``TimeSeriesSplit``. Instead of a handful of expanding folds, WFV
     slides a fixed train/test window forward through history one step at a
     time, exactly mirroring how a trading desk periodically re-fits a model:

         |■■■■■■■■■■■■ train (12m) ■■■■■■■■■■■■|░ test 1m ░|
                  |■■■■■■■■■■■■ train (12m) ■■■■■■■■■■■■|░ test 1m ░|
                           |■■■■■■■■■■■■ train (12m) ■■■■■■■■■■■■|░ test 1m ░|

     Every test window lies strictly AFTER its train window, so look-ahead bias
     is structurally impossible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

import config
from src.utils import get_logger

logger = get_logger("data_splitting")


# ===========================================================================
# Final hold-out split
# ===========================================================================
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


# ===========================================================================
# Walk-Forward Validation cross-validator
# ===========================================================================
def _n_samples(X) -> int:
    """Number of rows in X, accepting DataFrame / ndarray / list."""
    if hasattr(X, "shape"):
        return int(X.shape[0])
    return len(X)


class WalkForwardValidator:
    """
    Rolling (or anchored) walk-forward splitter compatible with scikit-learn's
    cross-validation API (``split`` + ``get_n_splits``), so it drops straight
    into ``GridSearchCV(cv=...)``.

    Parameters
    ----------
    train_size : number of samples in each training window.
    test_size  : number of samples in each (out-of-sample) test window.
    step       : how many samples to advance between folds. Defaults to
                 ``test_size`` (non-overlapping, contiguous test windows).
    rolling    : if True, the train window is a FIXED-length block that rolls
                 forward (drops the oldest data). If False, the train window is
                 ANCHORED at the start and expands.
    max_splits : optionally keep only the most-recent ``max_splits`` folds —
                 used to bound the cost of hyper-parameter search while still
                 validating on the most regime-relevant history.
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        step: int | None = None,
        rolling: bool = True,
        max_splits: int | None = None,
    ) -> None:
        if train_size <= 0 or test_size <= 0:
            raise ValueError("train_size and test_size must be positive.")
        self.train_size = int(train_size)
        self.test_size = int(test_size)
        self.step = int(step) if step else int(test_size)
        self.rolling = rolling
        self.max_splits = max_splits

    # -- internal: compute the train-end index of every fold ----------------
    def _fold_train_ends(self, n: int) -> list[int]:
        ends: list[int] = []
        train_end = self.train_size
        while train_end + self.test_size <= n:
            ends.append(train_end)
            train_end += self.step
        if self.max_splits is not None and len(ends) > self.max_splits:
            ends = ends[-self.max_splits :]  # keep most recent folds
        return ends

    # -- scikit-learn CV API ------------------------------------------------
    def split(
        self, X, y=None, groups=None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = _n_samples(X)
        ends = self._fold_train_ends(n)
        if not ends:
            raise ValueError(
                f"Not enough samples ({n}) for WFV with train_size="
                f"{self.train_size} + test_size={self.test_size}. "
                "Reduce the window sizes in WalkForwardConfig."
            )
        for train_end in ends:
            train_start = 0 if not self.rolling else max(0, train_end - self.train_size)
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(train_end, train_end + self.test_size)
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if X is None:
            raise ValueError("WalkForwardValidator.get_n_splits requires X.")
        return len(self._fold_train_ends(_n_samples(X)))


def make_walk_forward_cv(
    cfg: config.WalkForwardConfig = config.WALK_FORWARD,
    cap_splits: bool = True,
) -> WalkForwardValidator:
    """
    Build the project's standard walk-forward cross-validator from config.

    Parameters
    ----------
    cap_splits : when True (the default, used for hyper-parameter search) the
        number of folds is capped at ``cfg.cv_max_splits``. Pass False to get
        the full step-by-step fold sequence (e.g. for diagnostic backtests).
    """
    return WalkForwardValidator(
        train_size=cfg.train_size,
        test_size=cfg.test_size,
        step=cfg.step_size,
        rolling=cfg.rolling,
        max_splits=cfg.cv_max_splits if cap_splits else None,
    )


def describe_walk_forward(
    n_samples: int, cfg: config.WalkForwardConfig = config.WALK_FORWARD
) -> pd.DataFrame:
    """
    Return a human-readable table of the WFV fold geometry for a given sample
    count — useful for logging/sanity-checking before a long training run.
    """
    wfv = make_walk_forward_cv(cfg, cap_splits=True)
    rows = []
    dummy = np.empty((n_samples, 1))
    for i, (tr, te) in enumerate(wfv.split(dummy)):
        rows.append(
            {
                "fold": i,
                "train_start": int(tr[0]),
                "train_end": int(tr[-1]),
                "test_start": int(te[0]),
                "test_end": int(te[-1]),
                "n_train": len(tr),
                "n_test": len(te),
            }
        )
    return pd.DataFrame(rows)
