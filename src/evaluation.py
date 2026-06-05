"""
src/evaluation.py  —  CRISP-DM PHASE 5 (Evaluation: predictive performance)
===========================================================================
Turns raw model predictions on the hold-out test set into the metrics and
plots an investment committee would actually scrutinise.

Regression (next-day log return)
    * MAE, RMSE — average / penalised error magnitude
    * R² — variance explained (often near 0 for daily returns; that's normal)
    * Directional accuracy — % of days the SIGN of the prediction is correct,
      the metric that actually matters for a trading signal.

Classification (next-day direction)
    * Confusion matrix
    * Accuracy, Precision, Recall, F1
    * ROC-AUC + ROC curve
    * "UP-day precision" — when the model says UP, how often is it right
      (this drives the long-only entry decision).

All figures are written to ``outputs/figures`` (headless-safe).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config
from src.modeling import TrainedModel, predict_proba, predict_regression
from src.utils import banner, get_logger

logger = get_logger("evaluation")


# ===========================================================================
# Regression evaluation
# ===========================================================================
@dataclass
class RegressionReport:
    ticker: str
    mae: float
    rmse: float
    r2: float
    directional_accuracy: float
    n: int


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of observations where sign(prediction) == sign(actual)."""
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    # Treat exact zeros as "up" to avoid sign(0)=0 mismatches.
    true_sign[true_sign == 0] = 1
    pred_sign[pred_sign == 0] = 1
    return float(np.mean(true_sign == pred_sign))


def evaluate_regression(
    model: TrainedModel, X_test: pd.DataFrame, y_test: pd.Series
) -> RegressionReport:
    """Compute regression metrics on the hold-out set."""
    y_pred = predict_regression(model, X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    report = RegressionReport(
        ticker=model.ticker,
        mae=float(mean_absolute_error(y_test, y_pred)),
        rmse=rmse,
        r2=float(r2_score(y_test, y_pred)),
        directional_accuracy=directional_accuracy(y_test.to_numpy(), y_pred),
        n=len(y_test),
    )
    logger.info(
        "[%s] REGRESSION  MAE=%.6f RMSE=%.6f R²=%.4f DirAcc=%.3f",
        report.ticker,
        report.mae,
        report.rmse,
        report.r2,
        report.directional_accuracy,
    )
    return report


# ===========================================================================
# Classification evaluation
# ===========================================================================
@dataclass
class ClassificationReport:
    ticker: str
    accuracy: float
    precision_up: float
    recall_up: float
    f1: float
    roc_auc: float
    confusion: list[list[int]]  # [[TN, FP], [FN, TP]]
    n: int


def evaluate_classification(
    model: TrainedModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> ClassificationReport:
    """Compute classification metrics + confusion matrix on the hold-out set."""
    proba_up = predict_proba(model, X_test)
    y_pred = (proba_up >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = ClassificationReport(
        ticker=model.ticker,
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision_up=float(precision_score(y_test, y_pred, zero_division=0)),
        recall_up=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, proba_up)),
        confusion=cm.tolist(),
        n=len(y_test),
    )
    logger.info(
        "[%s] CLASSIFY  Acc=%.3f Prec(UP)=%.3f Rec(UP)=%.3f F1=%.3f AUC=%.3f",
        report.ticker,
        report.accuracy,
        report.precision_up,
        report.recall_up,
        report.f1,
        report.roc_auc,
    )
    return report


# ===========================================================================
# Plots
# ===========================================================================
def plot_confusion_matrix(
    report: ClassificationReport, filename: str | None = None
) -> str:
    """Render the confusion matrix as an annotated heatmap."""
    config.ensure_dirs()
    cm = np.array(report.confusion)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["DOWN (0)", "UP (1)"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{report.ticker} — Confusion Matrix")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13,
            )
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = config.FIGURE_DIR / (filename or f"{report.ticker}_confusion.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", path)
    return str(path)


def plot_roc_curve(
    model: TrainedModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    filename: str | None = None,
) -> str:
    """Render the ROC curve with the AUC annotated."""
    config.ensure_dirs()
    proba_up = predict_proba(model, X_test)
    fpr, tpr, _ = roc_curve(y_test, proba_up)
    auc = roc_auc_score(y_test, proba_up)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model.ticker} — ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = config.FIGURE_DIR / (filename or f"{model.ticker}_roc.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Saved ROC curve → %s", path)
    return str(path)


# ===========================================================================
# Aggregated report
# ===========================================================================
def evaluation_summary(
    reg_reports: list[RegressionReport],
    clf_reports: list[ClassificationReport],
) -> pd.DataFrame:
    """Combine per-ticker regression + classification metrics into one table."""
    reg = pd.DataFrame([asdict(r) for r in reg_reports]).set_index("ticker")
    clf = pd.DataFrame([asdict(c) for c in clf_reports]).drop(
        columns=["confusion"]
    ).set_index("ticker")
    summary = reg.join(clf, lsuffix="_reg", rsuffix="_clf")

    print(banner("PHASE 5 — MODEL EVALUATION SUMMARY"))
    print(summary.round(4).to_string())

    config.ensure_dirs()
    out = config.REPORT_DIR / "model_evaluation_summary.csv"
    summary.to_csv(out)
    logger.info("Saved evaluation summary → %s", out)
    return summary
