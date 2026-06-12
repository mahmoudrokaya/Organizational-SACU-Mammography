# tests/test_metrics.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from organizational_sacu_mammography.evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_statistics,
    compute_metrics_for_models,
    summarize_metrics,
    export_confusion_matrix_table,
)


def test_compute_confusion_statistics_correct_counts() -> None:
    y_true = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    y_pred = np.array(
        [
            0,
            1,
            1,
            0,
        ]
    )

    stats = compute_confusion_statistics(
        y_true,
        y_pred,
    )

    assert stats["true_negative"] == 1
    assert stats["false_positive"] == 1
    assert stats["true_positive"] == 1
    assert stats["false_negative"] == 1
    assert stats["sensitivity"] == 0.5
    assert stats["specificity"] == 0.5


def test_compute_classification_metrics_returns_expected_keys() -> None:
    y_true = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    y_prob = np.array(
        [
            0.10,
            0.40,
            0.70,
            0.90,
        ]
    )

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_probability=y_prob,
        threshold=0.50,
        model_name="TestModel",
    )

    required = {
        "model",
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "roc_auc",
        "average_precision",
        "sensitivity",
        "specificity",
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
    }

    assert required.issubset(metrics.keys())
    assert metrics["model"] == "TestModel"
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_metrics_for_multiple_models() -> None:
    y_true = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    probability_df = pd.DataFrame(
        {
            "ModelA": [
                0.10,
                0.20,
                0.80,
                0.90,
            ],
            "ModelB": [
                0.40,
                0.60,
                0.45,
                0.70,
            ],
        }
    )

    metrics_df = compute_metrics_for_models(
        y_true=y_true,
        probability_df=probability_df,
        threshold=0.50,
    )

    assert len(metrics_df) == 2
    assert "model" in metrics_df.columns
    assert "roc_auc" in metrics_df.columns


def test_summarize_metrics_orders_by_auc() -> None:
    y_true = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    probability_df = pd.DataFrame(
        {
            "StrongModel": [
                0.10,
                0.20,
                0.80,
                0.90,
            ],
            "WeakModel": [
                0.90,
                0.80,
                0.20,
                0.10,
            ],
        }
    )

    metrics_df = compute_metrics_for_models(
        y_true=y_true,
        probability_df=probability_df,
        threshold=0.50,
    )

    summary = summarize_metrics(
        metrics_df,
        sort_by="roc_auc",
    )

    assert summary.iloc[0]["model"] == "StrongModel"


def test_export_confusion_matrix_table() -> None:
    y_true = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    probability_df = pd.DataFrame(
        {
            "ModelA": [
                0.10,
                0.20,
                0.80,
                0.90,
            ],
        }
    )

    table = export_confusion_matrix_table(
        y_true=y_true,
        probability_df=probability_df,
        threshold=0.50,
    )

    assert len(table) == 1
    assert table.iloc[0]["true_positive"] == 2
    assert table.iloc[0]["true_negative"] == 2


def test_metrics_reject_inconsistent_lengths() -> None:
    y_true = np.array(
        [
            0,
            1,
        ]
    )

    y_prob = np.array(
        [
            0.10,
            0.50,
            0.90,
        ]
    )

    with pytest.raises(ValueError):
        compute_classification_metrics(
            y_true=y_true,
            y_probability=y_prob,
        )


def test_metrics_reject_nan_probabilities() -> None:
    y_true = np.array(
        [
            0,
            1,
        ]
    )

    y_prob = np.array(
        [
            0.10,
            np.nan,
        ]
    )

    with pytest.raises(ValueError):
        compute_classification_metrics(
            y_true=y_true,
            y_probability=y_prob,
        )
