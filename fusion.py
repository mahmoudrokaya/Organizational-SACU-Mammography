# src/organizational_sacu_mammography/models/fusion.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class FusionConfig:
    """
    Configuration for SACU meta-fusion.

    The fusion layer combines role-conditioned SACU probabilities and,
    optionally, adaptive coordination outputs into a final diagnostic score.
    """

    random_seed: int = 42
    probability_threshold: float = 0.50
    logistic_c: float = 1.0
    logistic_max_iter: int = 5000
    include_coordinated_probability: bool = True
    include_adaptive_weights: bool = True


def _validate_fusion_inputs(
    X_fusion: pd.DataFrame,
    y: pd.Series | np.ndarray,
) -> None:
    if not isinstance(X_fusion, pd.DataFrame):
        raise TypeError("X_fusion must be a pandas DataFrame.")

    if X_fusion.empty:
        raise ValueError("Fusion feature matrix is empty.")

    if len(X_fusion) != len(y):
        raise ValueError("X_fusion and y have inconsistent lengths.")

    if X_fusion.isna().any().any():
        raise ValueError("Fusion feature matrix contains NaN values.")

    if np.isinf(X_fusion.to_numpy()).any():
        raise ValueError("Fusion feature matrix contains infinite values.")


def build_fusion_feature_matrix(
    agent_probabilities: pd.DataFrame,
    coordinated_probability: Optional[pd.Series] = None,
    adaptive_weights: Optional[pd.DataFrame] = None,
    config: FusionConfig = FusionConfig(),
) -> pd.DataFrame:
    """
    Build the meta-fusion matrix from SACU outputs.
    """

    if agent_probabilities.empty:
        raise ValueError("agent_probabilities is empty.")

    frames = [agent_probabilities.copy()]

    if config.include_coordinated_probability:
        if coordinated_probability is None:
            raise ValueError(
                "coordinated_probability is required when "
                "include_coordinated_probability=True."
            )

        frames.append(
            coordinated_probability.rename(
                "coordinated_probability"
            ).to_frame()
        )

    if config.include_adaptive_weights:
        if adaptive_weights is None:
            raise ValueError(
                "adaptive_weights is required when "
                "include_adaptive_weights=True."
            )

        frames.append(adaptive_weights.copy())

    X_fusion = pd.concat(frames, axis=1)

    if X_fusion.isna().any().any():
        raise ValueError("Fusion matrix contains NaN values after concatenation.")

    return X_fusion


def train_fusion_model(
    X_fusion_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    config: FusionConfig = FusionConfig(),
) -> Pipeline:
    """
    Train learned shallow meta-fusion model.
    """

    _validate_fusion_inputs(
        X_fusion_train,
        y_train,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=config.logistic_c,
                    max_iter=config.logistic_max_iter,
                    random_state=config.random_seed,
                ),
            ),
        ]
    )

    model.fit(
        X_fusion_train,
        y_train,
    )

    return model


def predict_fusion_probabilities(
    fusion_model: Pipeline,
    X_fusion: pd.DataFrame,
) -> pd.Series:
    """
    Predict final SACU fusion probabilities.
    """

    if X_fusion.empty:
        raise ValueError("X_fusion is empty.")

    probabilities = fusion_model.predict_proba(
        X_fusion
    )[:, 1]

    return pd.Series(
        probabilities,
        index=X_fusion.index,
        name="sacu_fusion_probability",
    )


def evaluate_fusion_model(
    fusion_model: Pipeline,
    X_fusion: pd.DataFrame,
    y: pd.Series | np.ndarray,
    config: FusionConfig = FusionConfig(),
) -> Dict[str, float]:
    """
    Evaluate learned SACU fusion model.
    """

    probabilities = predict_fusion_probabilities(
        fusion_model,
        X_fusion,
    )

    predictions = (
        probabilities >= config.probability_threshold
    ).astype(int)

    y_true = np.asarray(y)

    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "mcc": matthews_corrcoef(
            y_true,
            predictions,
        ),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(
            y_true,
            probabilities,
        )
    except Exception:
        metrics["roc_auc"] = np.nan

    try:
        metrics["average_precision"] = average_precision_score(
            y_true,
            probabilities,
        )
    except Exception:
        metrics["average_precision"] = np.nan

    return metrics


def extract_fusion_coefficients(
    fusion_model: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extract coefficients from learned logistic fusion model.
    """

    classifier = fusion_model.named_steps["classifier"]

    coefficients = classifier.coef_.ravel()

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "absolute_coefficient": np.abs(coefficients),
            }
        )
        .sort_values(
            "absolute_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def summarize_fusion_model(
    evaluation_metrics: Dict[str, float],
    fusion_coefficients: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Summarize SACU fusion performance and learned contribution structure.
    """

    rows = []

    for metric, value in evaluation_metrics.items():
        rows.append(
            {
                "section": "performance",
                "metric": metric,
                "value": value,
            }
        )

    if fusion_coefficients is not None and not fusion_coefficients.empty:
        top = fusion_coefficients.iloc[0]

        rows.append(
            {
                "section": "fusion_structure",
                "metric": "top_fusion_feature",
                "value": top["feature"],
            }
        )

        rows.append(
            {
                "section": "fusion_structure",
                "metric": "top_absolute_coefficient",
                "value": float(top["absolute_coefficient"]),
            }
        )

    return pd.DataFrame(rows)


def export_fusion_predictions(
    probabilities: pd.Series,
    y_true: Optional[pd.Series | np.ndarray] = None,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Export final SACU fusion predictions.
    """

    output = pd.DataFrame(
        {
            "sacu_fusion_probability": probabilities,
            "sacu_fusion_prediction": (
                probabilities >= threshold
            ).astype(int),
        },
        index=probabilities.index,
    )

    if y_true is not None:
        output["true_label"] = np.asarray(y_true)

    return output.reset_index(drop=False)
