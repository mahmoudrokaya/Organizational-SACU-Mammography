# src/organizational_sacu_mammography/evaluation/metrics.py
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

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
)


def compute_confusion_statistics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> Dict[str, float]:
    """
    Compute binary confusion-matrix statistics.
    """

    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true and y_pred have inconsistent lengths.")

    tn, fp, fn, tp = confusion_matrix(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "positive_predictive_value": float(ppv),
        "negative_predictive_value": float(npv),
    }


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_probability: np.ndarray | pd.Series,
    threshold: float = 0.50,
    model_name: Optional[str] = None,
) -> Dict[str, float | str]:
    """
    Compute binary classification metrics from probabilities.
    """

    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_probability).astype(float)

    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError("y_true and y_probability have inconsistent lengths.")

    if np.isnan(y_prob_arr).any():
        raise ValueError("y_probability contains NaN values.")

    if np.isinf(y_prob_arr).any():
        raise ValueError("y_probability contains infinite values.")

    y_pred = (y_prob_arr >= threshold).astype(int)

    metrics: Dict[str, float | str] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "precision": float(
            precision_score(y_true_arr, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_arr, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true_arr, y_pred, zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true_arr, y_pred)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
    except Exception:
        metrics["roc_auc"] = np.nan

    try:
        metrics["average_precision"] = float(
            average_precision_score(y_true_arr, y_prob_arr)
        )
    except Exception:
        metrics["average_precision"] = np.nan

    metrics.update(
        compute_confusion_statistics(
            y_true_arr,
            y_pred,
        )
    )

    if model_name is not None:
        metrics["model"] = model_name

    return metrics


def compute_metrics_for_models(
    y_true: np.ndarray | pd.Series,
    probability_df: pd.DataFrame,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Compute metrics for multiple model probability columns.
    """

    if probability_df.empty:
        raise ValueError("probability_df is empty.")

    rows: List[Dict[str, float | str]] = []

    for column in probability_df.columns:
        rows.append(
            compute_classification_metrics(
                y_true=y_true,
                y_probability=probability_df[column],
                threshold=threshold,
                model_name=column,
            )
        )

    return pd.DataFrame(rows)


def summarize_metrics(
    metrics_df: pd.DataFrame,
    sort_by: str = "roc_auc",
) -> pd.DataFrame:
    """
    Produce a compact evaluation summary table.
    """

    if metrics_df.empty:
        return pd.DataFrame()

    if sort_by not in metrics_df.columns:
        raise KeyError(f"sort_by column not found: {sort_by}")

    cols = [
        "model",
        "threshold",
        "roc_auc",
        "average_precision",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "mcc",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    ]

    available_cols = [c for c in cols if c in metrics_df.columns]

    return (
        metrics_df[available_cols]
        .sort_values(sort_by, ascending=False)
        .reset_index(drop=True)
    )


def export_confusion_matrix_table(
    y_true: np.ndarray | pd.Series,
    probability_df: pd.DataFrame,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Export confusion-matrix counts for each model.
    """

    rows = []

    for column in probability_df.columns:
        y_pred = (
            probability_df[column].to_numpy(dtype=float) >= threshold
        ).astype(int)

        stats = compute_confusion_statistics(
            y_true,
            y_pred,
        )

        stats["model"] = column
        stats["threshold"] = float(threshold)

        rows.append(stats)

    return pd.DataFrame(rows)
