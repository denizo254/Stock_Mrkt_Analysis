"""
tests/test_data_splitting.py
============================
The leakage-critical module. These tests prove that **no validation fold ever
sees the future**, which is the single most important correctness property of
a financial ML pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_splitting import (
    WalkForwardValidator,
    chronological_split,
    make_walk_forward_cv,
)


# ---------------------------------------------------------------------------
# Walk-forward validator
# ---------------------------------------------------------------------------
def test_wfv_test_window_is_strictly_after_train_window():
    """Every fold: the test block begins exactly one step after train ends."""
    X = np.empty((200, 3))
    wfv = WalkForwardValidator(train_size=100, test_size=20, step=20, rolling=True)
    folds = list(wfv.split(X))
    assert len(folds) > 0
    for train_idx, test_idx in folds:
        # No overlap and correct ordering — the anti-leakage invariant.
        assert train_idx.max() < test_idx.min()
        assert test_idx.min() == train_idx.max() + 1
        assert len(test_idx) == 20


def test_wfv_rolling_window_is_fixed_length():
    X = np.empty((300, 2))
    wfv = WalkForwardValidator(train_size=120, test_size=30, step=30, rolling=True)
    for train_idx, _ in wfv.split(X):
        assert len(train_idx) == 120  # fixed rolling window drops oldest data


def test_wfv_anchored_window_expands():
    X = np.empty((300, 2))
    wfv = WalkForwardValidator(train_size=120, test_size=30, step=30, rolling=False)
    lengths = [len(tr) for tr, _ in wfv.split(X)]
    assert lengths[0] == 120
    assert lengths == sorted(lengths)          # monotonically non-decreasing
    assert lengths[-1] > lengths[0]            # genuinely expanded
    # Anchored => always starts at index 0.
    for train_idx, _ in wfv.split(X):
        assert train_idx.min() == 0


def test_wfv_consecutive_test_windows_do_not_overlap():
    X = np.empty((250, 1))
    wfv = WalkForwardValidator(train_size=100, test_size=25, step=25)
    test_blocks = [te for _, te in wfv.split(X)]
    for earlier, later in zip(test_blocks, test_blocks[1:]):
        assert earlier.max() < later.min()


def test_wfv_max_splits_keeps_most_recent_folds():
    X = np.empty((400, 1))
    capped = WalkForwardValidator(100, 20, step=20, max_splits=3)
    uncapped = WalkForwardValidator(100, 20, step=20)
    capped_folds = list(capped.split(X))
    uncapped_folds = list(uncapped.split(X))
    assert len(capped_folds) == 3
    assert capped.get_n_splits(X) == 3
    # The capped folds must be the LAST ones (closest to the present).
    assert capped_folds[-1][1].max() == uncapped_folds[-1][1].max()


def test_wfv_raises_when_too_few_samples():
    wfv = WalkForwardValidator(train_size=100, test_size=20)
    with pytest.raises(ValueError):
        list(wfv.split(np.empty((50, 1))))


def test_make_walk_forward_cv_respects_config_geometry():
    cv = make_walk_forward_cv()  # built from config.WALK_FORWARD
    X = np.empty((1500, 5))
    folds = list(cv.split(X))
    # 12-month train (252) / 1-month test (21) by default.
    tr, te = folds[0]
    assert len(tr) == 252
    assert len(te) == 21
    assert tr.max() < te.min()


# ---------------------------------------------------------------------------
# Chronological hold-out split
# ---------------------------------------------------------------------------
def test_chronological_split_is_time_ordered(feature_matrix):
    from src.feature_engineering import feature_columns

    cols = feature_columns(feature_matrix)
    split = chronological_split(feature_matrix, cols, "target_logret", test_size=0.2)

    # Train must lie entirely before test in calendar time (no shuffle).
    assert split.X_train.index.max() < split.X_test.index.min()
    # No index leakage across the boundary.
    assert set(split.X_train.index).isdisjoint(set(split.X_test.index))
    # Sizes add up and the split ratio is honoured (±1 for rounding).
    assert len(split.X_train) + len(split.X_test) == len(feature_matrix)
    assert abs(len(split.X_test) - int(0.2 * len(feature_matrix))) <= 1
