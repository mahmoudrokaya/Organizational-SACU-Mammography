# src/organizational_sacu_mammography/evaluation/threshold_analysis.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, precision_recall_curve

from .metrics import compute_classification_metrics


@dataclass
class ThresholdAnalysisConfig:
    """
    Configuration for clinical operating-point analysis.

    This module supports the Stage2E-style analysis used to compare default,
    Youden, high-sensitivity, and high-specificity thresholds.
    """

    default_threshold: float = 0.50
    high_sensitivity_target: float = 0.90
    high_specificity_target: float = 0.90
    threshold_grid_size: int = 1001
    eps: float = 1e-8


def _validate_inputs(
    y_true: np.ndarray | pd.Series,
    y_probability: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_probability).astype(float)

    if len(y) != len(p):
        raise ValueError("y_true and y_probability have inconsistent lengths.")

    if len(y) == 0:
        raise ValueError("Input arrays are empty.")

    if np.isnan(p).any():
        raise ValueError("y_probability contains NaN values.")

    if np.isinf(p).any():
        raise ValueError("y_probability contains infinite values.")

    return y, p


def _threshold_grid(
    config: ThresholdAnalysisConfig,
) -> np.ndarray:
    return np.linspace(
        0.0,
        1.0,
        config.threshold_grid_size,
    )


def _metrics_over_thresholds(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    thresholds: Sequence[float],
    model_name: str,
) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []

    for threshold in thresholds:
        row = compute_classification_metrics(
            y_true=y_true,
            y_probability=y_probability,
            threshold=float(threshold),
            model_name=model_name,
        )
        rows.append(row)

    return pd.DataFrame(rows)


def find_youden_threshold(
    y_true: np.ndarray | pd.Series,
    y_probability: np.ndarray | pd.Series,
    config: ThresholdAnalysisConfig = ThresholdAnalysisConfig(),
) -> float:
    """
    Select threshold maximizing sensitivity + specificity - 1.
    """

    y, p = _validate_inputs(y_true, y_probability)

    grid = _threshold_grid(config)

    metrics_df = _metrics_over_thresholds(
        y,
        p,
        grid,
        model_name="model",
    )

    youden = (
        metrics_df["sensitivity"]
        + metrics_df["specificity"]
        - 1.0
    )

    idx = int(youden.idxmax())

    return float(metrics_df.loc[idx, "threshold"])


def find_high_sensitivity_threshold(
    y_true: np.ndarray | pd.Series,
    y_probability: np.ndarray | pd.Series,
    config: ThresholdAnalysisConfig = ThresholdAnalysisConfig(),
) -> float:
    """
    Select the highest threshold that still achieves target sensitivity.
    """

    y, p = _validate_inputs(y_true, y_probability)

    grid = _threshold_grid(config)

    metrics_df = _metrics_over_thresholds(
        y,
        p,
        grid,
        model_name="model",
    )

    eligible = metrics_df[
        metrics_df["sensitivity"] >= config.high_sensitivity_target
    ]

    if eligible.empty:
        idx = metrics_df["sensitivity"].idxmax()
        return float(metrics_df.loc[idx, "threshold"])

    return float(eligible["threshold"].max())


def find_high_specificity_threshold(
    y_true: np.ndarray | pd.Series,
    y_probability: np.ndarray | pd.Series,
    config: ThresholdAnalysisConfig = ThresholdAnalysisConfig(),
) -> float:
    """
    Select the lowest threshold that achieves target specificity.
    """

    y, p = _validate_inputs(y_true, y_probability)

    grid = _threshold_grid(config)

    metrics_df = _metrics_over_thresholds(
        y,
        p,
        grid,
        model_name="model",
    )

    eligible = metrics_df[
        metrics_df["specificity"] >= config.high_specificity_target
    ]

    if eligible.empty:
        idx = metrics_df["specificity"].idxmax()
        return float(metrics_df.loc[idx, "threshold"])

    return float(eligible["threshold"].min())


def selected_operating_points(
    y_true: np.ndarray | pd.Series,
    y_probability: np.ndarray | pd.Series,
    config: ThresholdAnalysisConfig = ThresholdAnalysisConfig(),
) -> Dict[str, float]:
    """
    Return all clinically meaningful operating thresholds.
    """

    return {
        "default_0_50": float(config.default_threshold),
        "youden": find_youden_threshold(
            y_true,
            y_probability,
            config,
        ),
        "high_sensitivity": find_high_sensitivity_threshold(
            y_true,
            y_probability,
            config,
        ),
        "high_specificity": find_high_specificity_threshold(
            y_true,
            y_probability,
            config,
        ),
    }


def analyze_operating_points(
    y_true: np.ndarray | pd.Series,
    probability_df: pd.DataFrame,
    config: ThresholdAnalysisConfig = ThresholdAnalysisConfig(),
) -> pd.DataFrame:
    """
    Evaluate default, Youden, high-sensitivity, and high-specificity thresholds
    for each model probability column.
    """

    if probability_df.empty:
        raise ValueError("probability_df is empty.")

    rows: List[Dict[str, float | str]] = []

    for model_name in probability_df.columns:
        y, p = _validate_inputs(
            y_true,
            probability_df[model_name],
        )

        thresholds = selected_operating_points(
            y,
            p,
            config,
        )

        for operating_point, threshold in thresholds.items():
            metrics = compute_classification_metrics(
                y_true=y,
                y_probability=p,
                threshold=threshold,
                model_name=model_name,
            )

            metrics["operating_point"] = operating_point
            rows.append(metrics)

    return pd.DataFrame(rows)


def build_threshold_comparison_table(
    y_true: np.ndarray | pd.Series,
    probability_df: pd.DataFrame,
    config: ThresholdAnalysisConfig = ThresholdAnalysisConfig(),
) -> pd.DataFrame:
    """
    Build dense threshold-comparison table for all configured models.
    """

    if probability_df.empty:
        raise ValueError("probability_df is empty.")

    rows = []

    grid = _threshold_grid(config)

    for model_name in probability_df.columns:
        y, p = _validate_inputs(
            y_true,
            probability_df[model_name],
        )

        model_metrics = _metrics_over_thresholds(
            y,
            p,
            grid,
            model_name=model_name,
        )

        rows.append(model_metrics)

    return pd.concat(
        rows,
        axis=0,
        ignore_index=True,
    )


def compute_roc_pr_summary(
    y_true: np.ndarray | pd.Series,
    probability_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Export ROC/PR curve summary points for plotting and audit.
    """

    rows: List[Dict[str, float | str]] = []

    for model_name in probability_df.columns:
        y, p = _validate_inputs(
            y_true,
            probability_df[model_name],
        )

        fpr, tpr, roc_thresholds = roc_curve(y, p)

        for i in range(len(fpr)):
            rows.append(
                {
                    "model": model_name,
                    "curve": "roc",
                    "x": float(fpr[i]),
                    "y": float(tpr[i]),
                    "threshold": float(roc_thresholds[i]),
                }
            )

        precision, recall, pr_thresholds = precision_recall_curve(y, p)

        padded_thresholds = np.append(pr_thresholds, np.nan)

        for i in range(len(precision)):
            rows.append(
                {
                    "model": model_name,
                    "curve": "precision_recall",
                    "x": float(recall[i]),
                    "y": float(precision[i]),
                    "threshold": float(padded_thresholds[i]),
                }
            )

    return pd.DataFrame(rows)


def summarize_threshold_analysis(
    operating_point_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize clinical operating-point analysis.
    """

    if operating_point_df.empty:
        return pd.DataFrame()

    rows = []

    for model_name, group in operating_point_df.groupby("model"):
        best_balanced = group.sort_values(
            "balanced_accuracy",
            ascending=False,
        ).iloc[0]

        highest_sensitivity = group.sort_values(
            "sensitivity",
            ascending=False,
        ).iloc[0]

        highest_specificity = group.sort_values(
            "specificity",
            ascending=False,
        ).iloc[0]

        rows.extend(
            [
                {
                    "model": model_name,
                    "summary_metric": "best_balanced_accuracy_operating_point",
                    "value": best_balanced["operating_point"],
                },
                {
                    "model": model_name,
                    "summary_metric": "best_balanced_accuracy",
                    "value": float(best_balanced["balanced_accuracy"]),
                },
                {
                    "model": model_name,
                    "summary_metric": "highest_sensitivity_operating_point",
                    "value": highest_sensitivity["operating_point"],
                },
                {
                    "model": model_name,
                    "summary_metric": "highest_sensitivity",
                    "value": float(highest_sensitivity["sensitivity"]),
                },
                {
                    "model": model_name,
                    "summary_metric": "highest_specificity_operating_point",
                    "value": highest_specificity["operating_point"],
                },
                {
                    "model": model_name,
                    "summary_metric": "highest_specificity",
                    "value": float(highest_specificity["specificity"]),
                },
            ]
        )

    return pd.DataFrame(rows)
