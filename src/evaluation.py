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


# ===========================================================================
# STEP 3 — FEATURE IMPORTANCE & EXPLAINABILITY (SHAP)
# ===========================================================================
# Goal: identify exactly which engineered features (RSI, volatility, beta,
# lags, …) carry the predictive weight versus which are noise. We provide two
# complementary views:
#   1. Native XGBoost importances (gain / weight / cover) — fast, model-internal.
#   2. SHAP values — game-theoretic per-feature attribution that is consistent
#      and locally accurate. Falls back gracefully to native gain if `shap`
#      is not installed.

# Lazy / optional dependency.
try:
    import shap  # type: ignore

    _HAS_SHAP = True
except ImportError:  # pragma: no cover
    _HAS_SHAP = False
    logger.info("shap not installed — explainability will use native XGBoost gain.")


def _underlying_booster(model: TrainedModel):
    """Return the raw fitted estimator (unwrapping the linear Pipeline)."""
    est = model.estimator
    if hasattr(est, "named_steps"):           # linear Pipeline fallback
        return est.named_steps["model"]
    return est


def native_xgb_importances(model: TrainedModel) -> pd.DataFrame:
    """
    Extract XGBoost importances by **gain**, **weight**, and **cover**.

    * gain   — average loss reduction when the feature is used in a split
               (the most faithful "predictive importance" measure).
    * weight — number of times the feature is used to split.
    * cover  — average number of samples affected by its splits.

    Returns a DataFrame indexed by feature, sorted by gain (descending).
    Works for the tree models; for the linear fallback it returns absolute
    coefficients under a single 'gain' column.
    """
    booster_est = _underlying_booster(model)
    names = model.feature_names

    if not hasattr(booster_est, "get_booster"):
        # Linear fallback: absolute standardised coefficients.
        imp = model.feature_importance()
        df = pd.DataFrame({"gain": imp.abs()}) if imp is not None else pd.DataFrame()
        return df

    booster = booster_est.get_booster()
    booster.feature_names = list(names)
    frames = {}
    for kind in ("gain", "weight", "cover"):
        scores = booster.get_score(importance_type=kind)
        frames[kind] = pd.Series(scores)
    df = pd.DataFrame(frames).reindex(names).fillna(0.0)
    df = df.sort_values("gain", ascending=False)
    return df


def compute_shap_importance(
    model: TrainedModel,
    X: pd.DataFrame,
    sample_size: int = config.EXPLAIN.shap_sample_size,
) -> pd.Series | None:
    """
    Mean absolute SHAP value per feature (global importance).

    A random sample of ``sample_size`` rows from ``X`` is used to keep the
    TreeExplainer fast. Returns ``None`` (and logs) if shap is unavailable or
    the estimator is not tree-based.
    """
    if not _HAS_SHAP:
        return None
    booster_est = _underlying_booster(model)
    if not hasattr(booster_est, "get_booster"):
        logger.info("[%s] SHAP skipped — not a tree model.", model.ticker)
        return None

    n = min(sample_size, len(X))
    X_sample = X.sample(n=n, random_state=config.EXPLAIN.random_state) if n < len(X) else X

    explainer = shap.TreeExplainer(booster_est)
    shap_values = explainer.shap_values(X_sample)
    # Binary classifiers may return a list (one array per class); take class 1.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs, index=model.feature_names).sort_values(ascending=False)


def plot_feature_importance(
    importance: pd.Series,
    title: str,
    filename: str,
    top_n: int = config.EXPLAIN.top_n_features,
    xlabel: str = "Importance",
) -> str:
    """Horizontal bar chart of the top-N most important features."""
    config.ensure_dirs()
    top = importance.head(top_n).iloc[::-1]  # reverse so largest is on top
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top))))
    ax.barh(top.index, top.values, color="#2c7fb8")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    path = config.FIGURE_DIR / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Saved feature-importance plot → %s", path)
    return str(path)


@dataclass
class ExplainabilityResult:
    ticker: str
    task: str
    method: str                     # "shap" | "native_gain"
    importance: pd.Series           # ranked feature importances
    native_table: pd.DataFrame      # gain/weight/cover (may be empty for linear)
    figure_path: str
    csv_path: str


def explain_model(
    model: TrainedModel,
    X: pd.DataFrame,
) -> ExplainabilityResult:
    """
    Full explainability pass for one trained model.

    Prefers SHAP (if installed and tree-based); otherwise uses native XGBoost
    gain. Exports a ranked CSV to ``outputs/reports`` and a bar plot to
    ``outputs/figures``, and logs the top drivers vs the noise floor.
    """
    native = native_xgb_importances(model)

    shap_imp = compute_shap_importance(model, X) if config.EXPLAIN.enable_shap else None
    if shap_imp is not None:
        method, importance = "shap", shap_imp
        xlabel = "mean(|SHAP value|)"
    else:
        method = "native_gain"
        importance = native["gain"] if "gain" in native else native.iloc[:, 0]
        xlabel = "XGBoost gain"

    fig_path = plot_feature_importance(
        importance,
        title=f"{model.ticker} {model.task} — feature importance ({method})",
        filename=f"{model.ticker}_{model.task}_importance.png",
        xlabel=xlabel,
    )

    config.ensure_dirs()
    csv_path = config.REPORT_DIR / f"{model.ticker}_{model.task}_importance.csv"
    export = importance.rename("importance").to_frame()
    if not native.empty:
        export = export.join(native, how="left")
    export.to_csv(csv_path)

    # Log the signal-vs-noise read.
    top3 = ", ".join(importance.head(3).index)
    bottom3 = ", ".join(importance.tail(3).index)
    logger.info(
        "[%s/%s] explainability (%s): top drivers = %s | noise floor = %s",
        model.ticker, model.task, method, top3, bottom3,
    )

    return ExplainabilityResult(
        ticker=model.ticker,
        task=model.task,
        method=method,
        importance=importance,
        native_table=native,
        figure_path=fig_path,
        csv_path=str(csv_path),
    )


# ===========================================================================
# ROBUSTNESS — honest walk-forward (rolling-retrain) model evaluation
# ===========================================================================
# The single train/test split above reports metrics on ONE hold-out period and
# from ONE fit. A more honest read re-trains the model as it walks forward and
# scores every prediction out-of-sample. We reuse the walk-forward generator
# from ``signals`` (each prediction trained only on prior data) and score the
# concatenated OOS predictions against the realised targets across the WHOLE
# timeline — a far less luck-dependent estimate of live performance.


@dataclass
class WalkForwardEval:
    ticker: str
    engine: str
    n_oos: int                 # number of out-of-sample predictions scored
    # regression (OOS)
    mae: float
    rmse: float
    directional_accuracy: float
    # classification (OOS), NaN if the classifier could not be scored
    accuracy: float
    roc_auc: float


def walk_forward_evaluation(
    long: "pd.DataFrame",
    ticker: str,
    engine: str = "linear",
    refit_freq: int = 21,
    lookback: int = 252,
) -> WalkForwardEval:
    """
    Honest, rolling-retrain out-of-sample evaluation for one ticker.

    Returns regression metrics (MAE / RMSE / directional accuracy) on the
    concatenated walk-forward predictions, plus classification metrics
    (accuracy / ROC-AUC) when the direction signal is available.
    """
    from src.feature_engineering import build_features_for_ticker
    from src.signals import walk_forward_predictions

    wf = walk_forward_predictions(
        long, ticker, engine=engine, refit_freq=refit_freq, lookback=lookback
    )
    feats = build_features_for_ticker(long, ticker)
    y_ret = feats["target_logret"].reindex(wf.index)
    y_dir = feats["target_dir"].reindex(wf.index)

    pred = wf["pred_logret"]
    rmse = float(np.sqrt(mean_squared_error(y_ret, pred)))
    diracc = directional_accuracy(y_ret.to_numpy(), pred.to_numpy())

    accuracy = roc_auc = float("nan")
    if "prob_up" in wf:
        proba = wf["prob_up"].dropna()
        common = proba.index.intersection(y_dir.index)
        if len(common) > 0 and y_dir.loc[common].nunique() == 2:
            yhat = (proba.loc[common] >= 0.5).astype(int)
            accuracy = float(accuracy_score(y_dir.loc[common], yhat))
            roc_auc = float(roc_auc_score(y_dir.loc[common], proba.loc[common]))

    result = WalkForwardEval(
        ticker=ticker,
        engine=engine,
        n_oos=len(wf),
        mae=float(mean_absolute_error(y_ret, pred)),
        rmse=rmse,
        directional_accuracy=diracc,
        accuracy=accuracy,
        roc_auc=roc_auc,
    )
    logger.info(
        "[%s] WALK-FORWARD OOS (n=%d): MAE=%.6f DirAcc=%.3f Acc=%.3f AUC=%.3f",
        ticker, result.n_oos, result.mae, result.directional_accuracy,
        result.accuracy, result.roc_auc,
    )
    return result


def walk_forward_eval_summary(
    long: "pd.DataFrame",
    tickers: list[str],
    engine: str = "linear",
) -> "pd.DataFrame":
    """Run :func:`walk_forward_evaluation` across tickers and tabulate/save it."""
    rows = [walk_forward_evaluation(long, t, engine=engine) for t in tickers]
    df = pd.DataFrame([asdict(r) for r in rows]).set_index("ticker")
    print(banner("ROBUSTNESS — WALK-FORWARD (ROLLING-RETRAIN) OOS METRICS"))
    print(df.round(4).to_string())
    config.ensure_dirs()
    df.to_csv(config.REPORT_DIR / "walk_forward_evaluation.csv")
    logger.info("Saved walk-forward eval → %s", config.REPORT_DIR / "walk_forward_evaluation.csv")
    return df
