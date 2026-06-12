r"""
Stage2C_Train_Baseline_Shallow_Models.py

Purpose
-------
Train transparent shallow baseline models on the Stage2B SACU-compatible
modeling matrices.

This stage is intentionally conservative and method-consistent:
1. Uses only Stage2B leakage-audited features.
2. Uses the official VinDr-Mammo train/test split.
3. Trains shallow / classical baselines before SACU.
4. Reports point metrics and confidence intervals.
5. Reports confusion matrices, ROC/PR curve data, calibration data,
   threshold analysis, and statistical comparison.
6. Does not introduce deep models or real longitudinal claims.

Models
------
1. Logistic Regression
2. Linear SVM
3. RBF SVM
4. Random Forest
5. Gradient Boosting

Inputs
------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\matrices\\Stage2B_X_train.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\matrices\\Stage2B_y_train.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\matrices\\Stage2B_X_test.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\matrices\\Stage2B_y_test.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\matrices\\Stage2B_train_metadata.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\matrices\\Stage2B_test_metadata.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\matrices\\Stage2B_Feature_Groups.json

Outputs
-------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2C_Baseline_Model_Performance.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2C_Baseline_Confusion_Matrices.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2C_Baseline_ROC_Curves.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2C_Baseline_PR_Curves.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2C_Baseline_Calibration_Curves.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2C_Baseline_Threshold_Analysis.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2C_Baseline_Statistical_Comparison.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\models\\Stage2C_*.joblib
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\reports\\Stage2C_Baseline_Model_Report.txt

Recommended save location
-------------------------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\scripts\\Stage2C_Train_Baseline_Shallow_Models.py
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(r"D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments")

MATRIX_DIR = PROJECT_ROOT / "matrices"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_REPORT_DIR = PROJECT_ROOT / "results" / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_REPORT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_X_TRAIN = MATRIX_DIR / "Stage2B_X_train.csv"
INPUT_Y_TRAIN = MATRIX_DIR / "Stage2B_y_train.csv"
INPUT_X_TEST = MATRIX_DIR / "Stage2B_X_test.csv"
INPUT_Y_TEST = MATRIX_DIR / "Stage2B_y_test.csv"
INPUT_TRAIN_META = MATRIX_DIR / "Stage2B_train_metadata.csv"
INPUT_TEST_META = MATRIX_DIR / "Stage2B_test_metadata.csv"
INPUT_FEATURE_GROUPS = MATRIX_DIR / "Stage2B_Feature_Groups.json"

OUTPUT_PERFORMANCE = RESULTS_TABLE_DIR / "Stage2C_Baseline_Model_Performance.csv"
OUTPUT_CONFUSION = RESULTS_TABLE_DIR / "Stage2C_Baseline_Confusion_Matrices.csv"
OUTPUT_ROC = RESULTS_TABLE_DIR / "Stage2C_Baseline_ROC_Curves.csv"
OUTPUT_PR = RESULTS_TABLE_DIR / "Stage2C_Baseline_PR_Curves.csv"
OUTPUT_CALIBRATION = RESULTS_TABLE_DIR / "Stage2C_Baseline_Calibration_Curves.csv"
OUTPUT_THRESHOLDS = RESULTS_TABLE_DIR / "Stage2C_Baseline_Threshold_Analysis.csv"
OUTPUT_STATS = RESULTS_TABLE_DIR / "Stage2C_Baseline_Statistical_Comparison.csv"
OUTPUT_PREDICTIONS = RESULTS_TABLE_DIR / "Stage2C_Baseline_Test_Predictions.csv"
OUTPUT_SUMMARY_JSON = RESULTS_TABLE_DIR / "Stage2C_Baseline_Model_Summary.json"
OUTPUT_REPORT = RESULTS_REPORT_DIR / "Stage2C_Baseline_Model_Report.txt"

RANDOM_SEED = 42
N_BOOTSTRAP = 1000
N_CALIBRATION_BINS = 10

POS_LABEL = 1
NEG_LABEL = 0


# =============================================================================
# Helpers
# =============================================================================

def safe_str(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path, low_memory=False)


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_y(df: pd.DataFrame) -> np.ndarray:
    if "target" in df.columns:
        return df["target"].astype(int).values
    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(int).values
    raise ValueError("Could not identify target column.")


def positive_scores(model, X: pd.DataFrame) -> np.ndarray:
    """
    Return positive-class scores/probabilities.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, -1]

    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        decision = np.asarray(decision)
        if decision.ndim > 1:
            decision = decision[:, -1]
        # Min-max transform decision scores for calibration-like outputs.
        d_min = np.min(decision)
        d_max = np.max(decision)
        if d_max - d_min < 1e-12:
            return np.full_like(decision, 0.5, dtype=float)
        return (decision - d_min) / (d_max - d_min)

    pred = model.predict(X)
    return pred.astype(float)


def threshold_predictions(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (scores >= threshold).astype(int)


def specificity_from_confusion(tn: int, fp: int) -> float:
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def sensitivity_from_confusion(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) > 0 else np.nan


def np_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


# =============================================================================
# Models
# =============================================================================

def build_models() -> Dict[str, object]:
    """
    Shallow/classical baselines.
    Class imbalance is handled using class_weight when supported.
    """

    models: Dict[str, object] = {}

    models["LogisticRegression"] = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=RANDOM_SEED,
    )

    # LinearSVC does not expose probabilities, so calibrate it on training folds.
    linear_svc = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=10000,
        random_state=RANDOM_SEED,
    )

    models["LinearSVM_Calibrated"] = CalibratedClassifierCV(
        estimator=linear_svc,
        method="sigmoid",
        cv=3,
    )

    models["RBFSVM"] = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_SEED,
    )

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )

    models["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_SEED,
    )

    return models


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = threshold_predictions(scores, threshold)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity_recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity_from_confusion(tn, fp),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else np.nan,
        "pr_auc": average_precision_score(y_true, scores) if len(np.unique(y_true)) == 2 else np.nan,
        "brier_score": brier_score_loss(y_true, scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return {k: np_float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v for k, v in out.items()}


def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric_name: str,
    threshold: float = 0.5,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        s_b = scores[idx]

        if len(np.unique(y_b)) < 2:
            continue

        try:
            metrics = compute_metrics(y_b, s_b, threshold)
            vals.append(metrics[metric_name])
        except Exception:
            continue

    if not vals:
        return np.nan, np.nan

    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def find_best_threshold_youden(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    youden = tpr - fpr
    idx = int(np.argmax(youden))

    thr = thresholds[idx]

    if np.isinf(thr) or np.isnan(thr):
        return 0.5

    return float(thr)


def build_threshold_analysis(model_name: str, y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows = []

    thresholds = sorted(set(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, find_best_threshold_youden(y_true, scores)]
    ))

    for thr in thresholds:
        metrics = compute_metrics(y_true, scores, thr)
        rows.append({
            "model": model_name,
            **metrics,
            "threshold_type": "youden" if abs(thr - find_best_threshold_youden(y_true, scores)) < 1e-12 else "fixed",
        })

    return pd.DataFrame(rows)


def build_curve_tables(model_name: str, y_true: np.ndarray, scores: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fpr, tpr, roc_thr = roc_curve(y_true, scores)
    precision, recall, pr_thr = precision_recall_curve(y_true, scores)

    roc_df = pd.DataFrame({
        "model": model_name,
        "fpr": fpr,
        "tpr_sensitivity": tpr,
        "threshold": roc_thr,
    })

    # PR thresholds length = len(precision)-1
    pr_thresholds = list(pr_thr) + [np.nan]

    pr_df = pd.DataFrame({
        "model": model_name,
        "precision": precision,
        "recall_sensitivity": recall,
        "threshold": pr_thresholds,
    })

    return roc_df, pr_df


def build_calibration_table(model_name: str, y_true: np.ndarray, scores: np.ndarray, n_bins: int = N_CALIBRATION_BINS) -> pd.DataFrame:
    df = pd.DataFrame({
        "y_true": y_true,
        "score": scores,
    })

    df["bin"] = pd.cut(
        df["score"],
        bins=np.linspace(0.0, 1.0, n_bins + 1),
        include_lowest=True,
        labels=False,
    )

    rows = []

    for b, g in df.groupby("bin", dropna=False):
        if pd.isna(b):
            continue

        rows.append({
            "model": model_name,
            "bin": int(b),
            "n": int(len(g)),
            "mean_predicted_probability": float(g["score"].mean()),
            "observed_positive_rate": float(g["y_true"].mean()),
        })

    return pd.DataFrame(rows)


def build_confusion_row(model_name: str, y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, object]:
    y_pred = threshold_predictions(scores, threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "model": model_name,
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# =============================================================================
# Statistical Comparisons
# =============================================================================

def permutation_test_auc_difference(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_permutations: int = 2000,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float]:
    """
    Paired permutation test for ROC-AUC difference.
    """
    rng = np.random.default_rng(seed)

    observed = roc_auc_score(y_true, scores_a) - roc_auc_score(y_true, scores_b)

    diffs = []

    for _ in range(n_permutations):
        swap = rng.random(len(y_true)) < 0.5

        perm_a = scores_a.copy()
        perm_b = scores_b.copy()

        perm_a[swap], perm_b[swap] = perm_b[swap], perm_a[swap]

        try:
            diff = roc_auc_score(y_true, perm_a) - roc_auc_score(y_true, perm_b)
            diffs.append(diff)
        except Exception:
            continue

    if not diffs:
        return observed, np.nan

    diffs = np.asarray(diffs)
    p = float(np.mean(np.abs(diffs) >= abs(observed)))

    return float(observed), p


def build_statistical_comparison(y_true: np.ndarray, prediction_scores: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    models = list(prediction_scores.keys())

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a = models[i]
            b = models[j]

            diff, p = permutation_test_auc_difference(
                y_true=y_true,
                scores_a=prediction_scores[a],
                scores_b=prediction_scores[b],
            )

            rows.append({
                "model_a": a,
                "model_b": b,
                "roc_auc_difference_a_minus_b": diff,
                "paired_permutation_p_value": p,
                "test": "paired_permutation_test_on_roc_auc",
                "n_permutations": 2000,
            })

    return pd.DataFrame(rows)


# =============================================================================
# Training
# =============================================================================

def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    models = build_models()

    performance_rows = []
    confusion_rows = []
    roc_frames = []
    pr_frames = []
    calibration_frames = []
    threshold_frames = []
    predictions = pd.DataFrame({"target": y_test})

    prediction_scores: Dict[str, np.ndarray] = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        start = time.perf_counter()

        model.fit(X_train, y_train)

        train_time = time.perf_counter() - start

        start_inf = time.perf_counter()
        scores = positive_scores(model, X_test)
        inference_time = time.perf_counter() - start_inf

        scores = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)

        prediction_scores[model_name] = scores
        predictions[f"{model_name}_score"] = scores
        predictions[f"{model_name}_pred_0p5"] = threshold_predictions(scores, 0.5)

        default_metrics = compute_metrics(y_test, scores, threshold=0.5)
        best_thr = find_best_threshold_youden(y_test, scores)
        best_metrics = compute_metrics(y_test, scores, threshold=best_thr)

        ci_metrics = {}
        for metric in ["roc_auc", "pr_auc", "balanced_accuracy", "sensitivity_recall", "specificity", "f1"]:
            lo, hi = bootstrap_ci(y_test, scores, metric_name=metric, threshold=0.5)
            ci_metrics[f"{metric}_ci95_low"] = lo
            ci_metrics[f"{metric}_ci95_high"] = hi

        performance_rows.append({
            "model": model_name,
            "train_records": int(len(y_train)),
            "test_records": int(len(y_test)),
            "test_positive": int((y_test == 1).sum()),
            "test_negative": int((y_test == 0).sum()),
            "train_time_sec": float(train_time),
            "test_inference_time_sec": float(inference_time),
            "test_inference_time_ms_per_exam": float(1000.0 * inference_time / max(len(y_test), 1)),
            **default_metrics,
            "youden_threshold": best_thr,
            "youden_accuracy": best_metrics["accuracy"],
            "youden_balanced_accuracy": best_metrics["balanced_accuracy"],
            "youden_sensitivity": best_metrics["sensitivity_recall"],
            "youden_specificity": best_metrics["specificity"],
            "youden_f1": best_metrics["f1"],
            **ci_metrics,
        })

        confusion_rows.append(build_confusion_row(model_name, y_test, scores, threshold=0.5))

        roc_df, pr_df = build_curve_tables(model_name, y_test, scores)
        roc_frames.append(roc_df)
        pr_frames.append(pr_df)

        calibration_frames.append(build_calibration_table(model_name, y_test, scores))
        threshold_frames.append(build_threshold_analysis(model_name, y_test, scores))

        model_path = MODEL_DIR / f"Stage2C_{model_name}.joblib"
        joblib.dump(model, model_path)

    performance_df = pd.DataFrame(performance_rows)
    confusion_df = pd.DataFrame(confusion_rows)
    roc_all = pd.concat(roc_frames, ignore_index=True) if roc_frames else pd.DataFrame()
    pr_all = pd.concat(pr_frames, ignore_index=True) if pr_frames else pd.DataFrame()
    calibration_all = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame()
    thresholds_all = pd.concat(threshold_frames, ignore_index=True) if threshold_frames else pd.DataFrame()
    stats_df = build_statistical_comparison(y_test, prediction_scores)

    return (
        performance_df,
        confusion_df,
        roc_all,
        pr_all,
        calibration_all,
        thresholds_all,
        stats_df,
        predictions,
    )


# =============================================================================
# Summary and Report
# =============================================================================

def save_json_summary(performance_df: pd.DataFrame) -> None:
    if performance_df.empty:
        best_auc_model = ""
        best_auc = np.nan
        best_bal_model = ""
        best_bal = np.nan
    else:
        best_auc_row = performance_df.sort_values("roc_auc", ascending=False).iloc[0]
        best_bal_row = performance_df.sort_values("balanced_accuracy", ascending=False).iloc[0]
        best_auc_model = safe_str(best_auc_row["model"])
        best_auc = float(best_auc_row["roc_auc"])
        best_bal_model = safe_str(best_bal_row["model"])
        best_bal = float(best_bal_row["balanced_accuracy"])

    data = {
        "generated": str(datetime.now()),
        "project_root": str(PROJECT_ROOT),
        "inputs": {
            "X_train": str(INPUT_X_TRAIN),
            "y_train": str(INPUT_Y_TRAIN),
            "X_test": str(INPUT_X_TEST),
            "y_test": str(INPUT_Y_TEST),
            "feature_groups": str(INPUT_FEATURE_GROUPS),
        },
        "best_models": {
            "best_roc_auc_model": best_auc_model,
            "best_roc_auc": best_auc,
            "best_balanced_accuracy_model": best_bal_model,
            "best_balanced_accuracy": best_bal,
        },
        "performance": performance_df.to_dict(orient="records"),
    }

    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_report(
    performance_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_groups: Dict,
) -> None:
    lines = []

    lines.append("=" * 100)
    lines.append("STAGE 2C BASELINE SHALLOW MODEL REPORT")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append("")

    lines.append("METHOD CONSISTENCY")
    lines.append("-" * 100)
    lines.append("Used Stage2B leakage-audited SACU-compatible feature matrix.")
    lines.append("Used official VinDr-Mammo training/test split.")
    lines.append("No real longitudinal variables were introduced.")
    lines.append("Trained shallow/classical baselines only.")
    lines.append("Reported ROC-AUC, PR-AUC, sensitivity, specificity, F1, MCC, confusion matrices, calibration, threshold analysis, and paired permutation tests.")
    lines.append("")

    lines.append("DATA SHAPE")
    lines.append("-" * 100)
    lines.append(f"X_train: {X_train.shape}")
    lines.append(f"X_test: {X_test.shape}")
    lines.append(f"Train positives: {int((y_train == 1).sum())}")
    lines.append(f"Train negatives: {int((y_train == 0).sum())}")
    lines.append(f"Test positives: {int((y_test == 1).sum())}")
    lines.append(f"Test negatives: {int((y_test == 0).sum())}")
    lines.append("")

    lines.append("FEATURE GROUPS")
    lines.append("-" * 100)
    for k, v in feature_groups.items():
        lines.append(f"{k}: {len(v)}")
    lines.append("")

    lines.append("MODEL PERFORMANCE")
    lines.append("-" * 100)
    if performance_df.empty:
        lines.append("No model performance available.")
    else:
        display_cols = [
            "model",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "sensitivity_recall",
            "specificity",
            "f1",
            "mcc",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "test_inference_time_ms_per_exam",
        ]
        lines.append(performance_df[display_cols].to_string(index=False))
    lines.append("")

    lines.append("CONFUSION MATRICES")
    lines.append("-" * 100)
    lines.append(confusion_df.to_string(index=False) if not confusion_df.empty else "No confusion matrices.")
    lines.append("")

    lines.append("STATISTICAL COMPARISON")
    lines.append("-" * 100)
    if stats_df.empty:
        lines.append("No statistical comparisons.")
    else:
        lines.append(stats_df.to_string(index=False))
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    for p in [
        OUTPUT_PERFORMANCE,
        OUTPUT_CONFUSION,
        OUTPUT_ROC,
        OUTPUT_PR,
        OUTPUT_CALIBRATION,
        OUTPUT_THRESHOLDS,
        OUTPUT_STATS,
        OUTPUT_PREDICTIONS,
        OUTPUT_SUMMARY_JSON,
        OUTPUT_REPORT,
    ]:
        lines.append(str(p))

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    warnings.filterwarnings("ignore")
    np.random.seed(RANDOM_SEED)

    print("=" * 100)
    print("STAGE 2C TRAIN BASELINE SHALLOW MODELS")
    print("=" * 100)
    print(f"Project root: {PROJECT_ROOT}")
    print("-" * 100)

    print("Loading matrices...")
    X_train = read_csv_required(INPUT_X_TRAIN)
    y_train_df = read_csv_required(INPUT_Y_TRAIN)
    X_test = read_csv_required(INPUT_X_TEST)
    y_test_df = read_csv_required(INPUT_Y_TEST)

    y_train = get_y(y_train_df)
    y_test = get_y(y_test_df)

    feature_groups = load_json(INPUT_FEATURE_GROUPS)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"Train positives: {int((y_train == 1).sum())}, negatives: {int((y_train == 0).sum())}")
    print(f"Test positives:  {int((y_test == 1).sum())}, negatives: {int((y_test == 0).sum())}")
    print("-" * 100)

    (
        performance_df,
        confusion_df,
        roc_df,
        pr_df,
        calibration_df,
        threshold_df,
        stats_df,
        predictions_df,
    ) = train_and_evaluate_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    print("Saving outputs...")
    performance_df.to_csv(OUTPUT_PERFORMANCE, index=False, encoding="utf-8-sig")
    confusion_df.to_csv(OUTPUT_CONFUSION, index=False, encoding="utf-8-sig")
    roc_df.to_csv(OUTPUT_ROC, index=False, encoding="utf-8-sig")
    pr_df.to_csv(OUTPUT_PR, index=False, encoding="utf-8-sig")
    calibration_df.to_csv(OUTPUT_CALIBRATION, index=False, encoding="utf-8-sig")
    threshold_df.to_csv(OUTPUT_THRESHOLDS, index=False, encoding="utf-8-sig")
    stats_df.to_csv(OUTPUT_STATS, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(OUTPUT_PREDICTIONS, index=False, encoding="utf-8-sig")

    save_json_summary(performance_df)

    write_report(
        performance_df=performance_df,
        confusion_df=confusion_df,
        stats_df=stats_df,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_groups=feature_groups,
    )

    print()
    print("STAGE 2C COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"Performance:             {OUTPUT_PERFORMANCE}")
    print(f"Confusion matrices:      {OUTPUT_CONFUSION}")
    print(f"ROC curves:              {OUTPUT_ROC}")
    print(f"PR curves:               {OUTPUT_PR}")
    print(f"Calibration curves:      {OUTPUT_CALIBRATION}")
    print(f"Threshold analysis:      {OUTPUT_THRESHOLDS}")
    print(f"Statistical comparison:  {OUTPUT_STATS}")
    print(f"Predictions:             {OUTPUT_PREDICTIONS}")
    print(f"JSON summary:            {OUTPUT_SUMMARY_JSON}")
    print(f"Text report:             {OUTPUT_REPORT}")
    print("=" * 100)


if __name__ == "__main__":
    main()