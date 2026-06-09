"""
src/modeling.py  —  CRISP-DM PHASE 4 (Modeling)
===============================================
A dual-model approach to next-day forecasting:

  1. **Regression** — predict the exact next-day log return.
       engine = "xgboost"  → XGBRegressor
       engine = "linear"   → Ridge (with feature standardisation)

  2. **Classification** — predict next-day direction (UP=1 / DOWN=0).
       engine = "xgboost"  → XGBClassifier
       engine = "linear"   → LogisticRegression (with standardisation)

Both tasks tune hyper-parameters with ``GridSearchCV`` over a
``TimeSeriesSplit`` (expanding window) so that every validation fold is
strictly in the future relative to its training fold — no look-ahead leakage.

Trained estimators are persisted to ``outputs/models`` via joblib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.data_splitting import SplitData, make_walk_forward_cv
from src.utils import get_logger, timed

logger = get_logger("modeling")

# XGBoost is optional-at-runtime: import lazily so the linear path still works
# even if the wheel is missing on an exotic platform.
try:
    from xgboost import XGBClassifier, XGBRegressor

    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False
    logger.warning("xgboost not importable — falling back to linear models only.")


# ===========================================================================
# Estimator + hyper-parameter grid factories
# ===========================================================================
def _regularised_xgb_grid() -> dict:
    """
    Heavily-regularised XGBoost search grid shared by both tasks (Step 1).

    Rationale — daily equity signals are dominated by noise, so we fight
    overfitting on every available axis:
      * ``max_depth`` ∈ {1,2,3}      — very shallow trees (weak learners).
      * ``reg_alpha`` (L1)            — drives weak feature weights to zero.
      * ``reg_lambda`` (L2)           — shrinks leaf weights smoothly.
      * ``gamma`` > 0                 — refuses splits that don't cut loss enough.
    ``subsample`` / ``colsample_bytree`` are FIXED at 0.7 on the estimator to
    force row/feature diversification, and are therefore not part of the grid.
    """
    m = config.MODEL
    return {
        "max_depth": list(m.max_depth_grid),
        "learning_rate": list(m.learning_rate_grid),
        "reg_alpha": list(m.reg_alpha_grid),
        "reg_lambda": list(m.reg_lambda_grid),
        "gamma": list(m.gamma_grid),
    }


def _xgb_common_kwargs() -> dict:
    """Fixed XGBoost constructor kwargs (structural regularisation)."""
    m = config.MODEL
    return dict(
        n_estimators=m.n_estimators,
        subsample=m.subsample,
        colsample_bytree=m.colsample_bytree,
        random_state=m.random_state,
        n_jobs=-1,
        tree_method="hist",
    )


def _build_regressor(engine: str) -> tuple[object, dict]:
    """Return an (estimator, param_grid) pair for the regression task."""
    if engine == "xgboost" and _HAS_XGB:
        est = XGBRegressor(objective="reg:squarederror", **_xgb_common_kwargs())
        return est, _regularised_xgb_grid()

    # Linear fallback: standardise then Ridge. A Pipeline keeps scaling inside
    # the CV loop so validation folds are never used to fit the scaler. Ridge's
    # alpha is itself an L2 penalty, so the linear path is regularised too.
    est = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    return est, grid


def _build_classifier(engine: str) -> tuple[object, dict]:
    """Return an (estimator, param_grid) pair for the classification task."""
    if engine == "xgboost" and _HAS_XGB:
        est = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **_xgb_common_kwargs(),
        )
        return est, _regularised_xgb_grid()

    est = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000, random_state=config.MODEL.random_state
                ),
            ),
        ]
    )
    grid = {"model__C": [0.01, 0.1, 1.0, 10.0]}
    return est, grid


# ===========================================================================
# Result container
# ===========================================================================
@dataclass
class TrainedModel:
    task: str                 # "regression" | "classification"
    ticker: str
    engine: str
    estimator: object         # fitted best estimator
    best_params: dict
    cv_score: float           # best mean CV score (sign per scoring metric)
    feature_names: list[str] = field(default_factory=list)

    def save(self) -> Path:
        config.ensure_dirs()
        path = config.MODEL_DIR / f"{self.ticker}_{self.task}.joblib"
        joblib.dump(self, path)
        logger.info("Saved %s model → %s", self.task, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "TrainedModel":
        return joblib.load(path)

    def feature_importance(self) -> pd.Series | None:
        """Return feature importances / coefficients if the engine exposes them."""
        est = self.estimator
        model = est.named_steps["model"] if isinstance(est, Pipeline) else est
        if hasattr(model, "feature_importances_"):
            vals = model.feature_importances_
        elif hasattr(model, "coef_"):
            vals = np.ravel(model.coef_)
        else:
            return None
        return pd.Series(vals, index=self.feature_names).sort_values(
            key=np.abs, ascending=False
        )


# ===========================================================================
# Training entry points
# ===========================================================================
def train_regressor(
    split: SplitData,
    ticker: str,
    engine: str = config.MODEL.regressor,
) -> TrainedModel:
    """Tune + fit the next-day log-return regressor on the training split."""
    est, grid = _build_regressor(engine)
    cv = make_walk_forward_cv()  # rolling walk-forward folds (Step 1)

    with timed(f"[{ticker}] regression WFV grid-search ({engine})", logger):
        search = GridSearchCV(
            estimator=est,
            param_grid=grid,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score=np.nan,  # a degenerate fold should not kill the search
        )
        search.fit(split.X_train, split.y_train)

    model = TrainedModel(
        task="regression",
        ticker=ticker,
        engine=engine if (engine == "xgboost" and _HAS_XGB) else "linear",
        estimator=search.best_estimator_,
        best_params=search.best_params_,
        cv_score=float(search.best_score_),  # negative MAE
        feature_names=list(split.X_train.columns),
    )
    logger.info(
        "[%s] best regression CV MAE=%.6f params=%s",
        ticker,
        -model.cv_score,
        model.best_params,
    )
    return model


def train_classifier(
    split: SplitData,
    ticker: str,
    engine: str = config.MODEL.classifier,
) -> TrainedModel:
    """Tune + fit the next-day direction classifier on the training split."""
    est, grid = _build_classifier(engine)
    cv = make_walk_forward_cv()  # rolling walk-forward folds (Step 1)

    with timed(f"[{ticker}] classification WFV grid-search ({engine})", logger):
        search = GridSearchCV(
            estimator=est,
            param_grid=grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score=np.nan,  # a single-class test fold should not abort tuning
        )
        search.fit(split.X_train, split.y_train)

    model = TrainedModel(
        task="classification",
        ticker=ticker,
        engine=engine if (engine == "xgboost" and _HAS_XGB) else "linear",
        estimator=search.best_estimator_,
        best_params=search.best_params_,
        cv_score=float(search.best_score_),  # ROC-AUC
        feature_names=list(split.X_train.columns),
    )
    logger.info(
        "[%s] best classification CV ROC-AUC=%.4f params=%s",
        ticker,
        model.cv_score,
        model.best_params,
    )
    return model


def predict_regression(model: TrainedModel, X: pd.DataFrame) -> np.ndarray:
    """Point predictions of next-day log return."""
    return model.estimator.predict(X)


def predict_proba(model: TrainedModel, X: pd.DataFrame) -> np.ndarray:
    """Predicted P(UP) for the classifier (probability of class 1)."""
    return model.estimator.predict_proba(X)[:, 1]
