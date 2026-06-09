"""
tests/test_modeling_evaluation.py
=================================
End-to-end smoke of the modeling + evaluation path on the LINEAR engine
(deterministic, fast, no xgboost/network required) plus exact checks of the
directional-accuracy metric.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data_splitting import chronological_split
from src.evaluation import (
    directional_accuracy,
    evaluate_classification,
    evaluate_regression,
)
from src.feature_engineering import feature_columns
from src.modeling import (
    predict_proba,
    predict_regression,
    train_classifier,
    train_regressor,
)


def test_directional_accuracy_exact():
    y_true = np.array([0.01, -0.02, 0.03, -0.01])
    y_pred = np.array([0.05, 0.01, 0.02, -0.04])  # signs: +,+,+,-  vs +,-,+,-
    # Matches on indices 0, 2, 3 -> 3/4.
    assert directional_accuracy(y_true, y_pred) == pytest.approx(0.75)


@pytest.fixture(scope="module")
def split(feature_matrix):
    cols = feature_columns(feature_matrix)
    return cols, feature_matrix


def test_regression_path_trains_and_scores(split):
    cols, fm = split
    s = chronological_split(fm, cols, "target_logret")
    model = train_regressor(s, "AAPL", engine="linear")
    preds = predict_regression(model, s.X_test)
    assert len(preds) == len(s.X_test)

    rep = evaluate_regression(model, s.X_test, s.y_test)
    assert rep.mae >= 0
    assert rep.rmse >= rep.mae - 1e-9          # RMSE >= MAE always
    assert 0.0 <= rep.directional_accuracy <= 1.0


def test_classification_path_trains_and_scores(split):
    cols, fm = split
    s = chronological_split(fm, cols, "target_dir")
    model = train_classifier(s, "AAPL", engine="linear")
    proba = predict_proba(model, s.X_test)
    assert ((proba >= 0) & (proba <= 1)).all()

    rep = evaluate_classification(model, s.X_test, s.y_test)
    assert 0.0 <= rep.accuracy <= 1.0
    assert 0.0 <= rep.roc_auc <= 1.0
    assert 0.0 <= rep.precision_up <= 1.0
    # Confusion matrix is 2x2 and its entries sum to the sample count.
    cm = np.array(rep.confusion)
    assert cm.shape == (2, 2)
    assert cm.sum() == len(s.y_test)


def test_feature_importance_extraction(split):
    """The linear model exposes coefficients -> importance Series is returned."""
    cols, fm = split
    s = chronological_split(fm, cols, "target_logret")
    model = train_regressor(s, "AAPL", engine="linear")
    imp = model.feature_importance()
    assert imp is not None
    assert set(imp.index) == set(model.feature_names)
