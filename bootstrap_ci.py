# src/organizational_sacu_mammography/evaluation/bootstrap_ci.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .metrics import compute_classification_metrics


@dataclass
class BootstrapCIConfig:
    """
    Configuration for bootstrap confidence intervals.
    """

    n_bootstrap: int = 2000
    confidence_level: float = 0.95
    threshold: float = 0.50
    random_seed: int = 42
    stratified: bool = True


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


def _bootstrap_indices(
    y_true: np.ndarray,
    rng: np.random.Generator,
    config: BootstrapCIConfig,
) -> np.ndarray:
    if not config.stratified:
        return rng.choice(
            np.arange(len(y_true)),
            size=len(y_true),
            replace=True,
        )

    positive_idx = np.where(y_true == 1)[0]
    negative_idx = np.where(y_true == 0)[0]

    if len(positive_idx) == 0 or len(negative_idx) == 0:
        return rng.choice(
            np.arange(len(y_true)),
            size=len(y_true),
            replace=True,
        )

    sampled_positive = rng.choice(
        positive_idx,
        size=len(positive_idx),
        replace=True,
    )

    sampled_negative = rng.choice(
        negative_idx,
        size=len(negative_idx),
        replace=True,
    )

    sampled = np.concatenate(
        [
            sampled_positive,
            sampled_negative,
        ]
    )

    rng.shuffle(sampled)

    return sampled


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray | pd.Series,
    probability_df: pd.DataFrame,
    metrics: Sequence[str] = (
        "roc_auc",
        "average_precision",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "mcc",
    ),
    config: BootstrapCIConfig = BootstrapCIConfig(),
) -> pd.DataFrame:
    """
    Compute bootstrap confidence intervals for one or more model outputs.
    """

    if probability_df.empty:
        raise ValueError("probability_df is empty.")

    rng = np.random.default_rng(config.random_seed)

    alpha = 1.0 - config.confidence_level
    lower_q = 100.0 * alpha / 2.0
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    rows: List[Dict[str, float | str | int]] = []

    for model_name in probability_df.columns:
        y, p = _validate_inputs(
            y_true,
            probability_df[model_name],
        )

        full_metrics = compute_classification_metrics(
            y_true=y,
            y_probability=p,
            threshold=config.threshold,
            model_name=model_name,
        )

        bootstrap_values: Dict[str, List[float]] = {
            metric: []
            for metric in metrics
        }

        for _ in range(config.n_bootstrap):
            idx = _bootstrap_indices(
                y,
                rng,
                config,
            )

            y_b = y[idx]
            p_b = p[idx]

            if len(np.unique(y_b)) < 2:
                continue

            sampled_metrics = compute_classification_metrics(
                y_true=y_b,
                y_probability=p_b,
                threshold=config.threshold,
                model_name=model_name,
            )

            for metric in metrics:
                value = sampled_metrics.get(metric, np.nan)

                if value is not None and not pd.isna(value):
                    bootstrap_values[metric].append(float(value))

        for metric in metrics:
            values = np.asarray(
                bootstrap_values[metric],
                dtype=float,
            )

            if values.size == 0:
                lower = np.nan
                upper = np.nan
                mean = np.nan
                std = np.nan
                n_valid = 0
            else:
                lower = float(np.percentile(values, lower_q))
                upper = float(np.percentile(values, upper_q))
                mean = float(np.mean(values))
                std = float(np.std(values))
                n_valid = int(values.size)

            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "point_estimate": full_metrics.get(metric, np.nan),
                    "bootstrap_mean": mean,
                    "bootstrap_std": std,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "confidence_level": float(config.confidence_level),
                    "n_bootstrap_requested": int(config.n_bootstrap),
                    "n_bootstrap_valid": n_valid,
                    "threshold": float(config.threshold),
                }
            )

    return pd.DataFrame(rows)


def compute_paired_bootstrap_difference(
    y_true: np.ndarray | pd.Series,
    probability_a: np.ndarray | pd.Series,
    probability_b: np.ndarray | pd.Series,
    model_a_name: str,
    model_b_name: str,
    metric: str = "roc_auc",
    config: BootstrapCIConfig = BootstrapCIConfig(),
) -> pd.DataFrame:
    """
    Compute paired bootstrap confidence interval for model A - model B.
    """

    y, p_a = _validate_inputs(y_true, probability_a)
    _, p_b = _validate_inputs(y_true, probability_b)

    rng = np.random.default_rng(config.random_seed)

    alpha = 1.0 - config.confidence_level
    lower_q = 100.0 * alpha / 2.0
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    full_a = compute_classification_metrics(
        y,
        p_a,
        threshold=config.threshold,
        model_name=model_a_name,
    )

    full_b = compute_classification_metrics(
        y,
        p_b,
        threshold=config.threshold,
        model_name=model_b_name,
    )

    point_difference = full_a[metric] - full_b[metric]

    differences: List[float] = []

    for _ in range(config.n_bootstrap):
        idx = _bootstrap_indices(
            y,
            rng,
            config,
        )

        y_b = y[idx]

        if len(np.unique(y_b)) < 2:
            continue

        metrics_a = compute_classification_metrics(
            y_b,
            p_a[idx],
            threshold=config.threshold,
            model_name=model_a_name,
        )

        metrics_b = compute_classification_metrics(
            y_b,
            p_b[idx],
            threshold=config.threshold,
            model_name=model_b_name,
        )

        if pd.isna(metrics_a.get(metric)) or pd.isna(metrics_b.get(metric)):
            continue

        differences.append(
            float(metrics_a[metric] - metrics_b[metric])
        )

    values = np.asarray(differences, dtype=float)

    return pd.DataFrame(
        [
            {
                "model_a": model_a_name,
                "model_b": model_b_name,
                "metric": metric,
                "point_difference": float(point_difference),
                "bootstrap_mean_difference": float(np.mean(values)) if values.size else np.nan,
                "ci_lower": float(np.percentile(values, lower_q)) if values.size else np.nan,
                "ci_upper": float(np.percentile(values, upper_q)) if values.size else np.nan,
                "n_bootstrap_requested": int(config.n_bootstrap),
                "n_bootstrap_valid": int(values.size),
                "confidence_level": float(config.confidence_level),
            }
        ]
    )


def summarize_bootstrap_results(
    ci_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce compact bootstrap-CI summary.
    """

    if ci_df.empty:
        return pd.DataFrame()

    rows = []

    for model_name, group in ci_df.groupby("model"):
        roc_auc = group[group["metric"] == "roc_auc"]

        if not roc_auc.empty:
            row = roc_auc.iloc[0]

            rows.append(
                {
                    "model": model_name,
                    "summary_metric": "roc_auc_ci",
                    "value": (
                        f"{row['point_estimate']:.4f} "
                        f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
                    ),
                }
            )

        balanced = group[group["metric"] == "balanced_accuracy"]

        if not balanced.empty:
            row = balanced.iloc[0]

            rows.append(
                {
                    "model": model_name,
                    "summary_metric": "balanced_accuracy_ci",
                    "value": (
                        f"{row['point_estimate']:.4f} "
                        f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
                    ),
                }
            )

    return pd.DataFrame(rows)
