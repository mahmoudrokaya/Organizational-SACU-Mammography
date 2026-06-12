# src/organizational_sacu_mammography/models/shallow_agent.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
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


SUPPORTED_AGENT_TYPES = {
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
}


@dataclass
class ShallowAgentConfig:
    """
    Configuration for an organizational shallow diagnostic agent.

    Each SACU role is implemented as an independent shallow learner
    operating on its own pathway-specific feature subset.
    """

    agent_name: str

    agent_type: str = "gradient_boosting"

    random_seed: int = 42

    probability_threshold: float = 0.50

    logistic_c: float = 1.0
    logistic_max_iter: int = 5000

    n_estimators: int = 300
    max_depth: Optional[int] = None

    learning_rate: float = 0.05

    feature_names: List[str] = field(default_factory=list)


def _validate_agent_config(
    config: ShallowAgentConfig,
) -> None:
    if config.agent_type not in SUPPORTED_AGENT_TYPES:
        raise ValueError(
            f"Unsupported agent type: {config.agent_type}"
        )

    if not config.agent_name:
        raise ValueError(
            "agent_name cannot be empty."
        )


def _build_estimator(
    config: ShallowAgentConfig,
):
    """
    Build the underlying shallow learner.
    """

    if config.agent_type == "logistic_regression":
        return Pipeline(
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

    if config.agent_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_seed,
            n_jobs=-1,
        )

    if config.agent_type == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_seed,
            n_jobs=-1,
        )

    if config.agent_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=(
                3
                if config.max_depth is None
                else config.max_depth
            ),
            random_state=config.random_seed,
        )

    raise ValueError(
        f"Unsupported agent type: {config.agent_type}"
    )


class ShallowDiagnosticAgent:
    """
    SACU organizational shallow agent.

    Examples:
    ---------
    LocalRegionalAgent
    MultiViewAgent
    BilateralAgent
    TemporalSpatialAgent
    MetadataAgent
    AdaptiveControlAgent
    """

    def __init__(
        self,
        config: ShallowAgentConfig,
    ) -> None:

        _validate_agent_config(config)

        self.config = config
        self.model = _build_estimator(config)

        self.is_fitted: bool = False

        self.training_metrics_: Dict[str, float] = {}
        self.feature_importance_: Optional[pd.DataFrame] = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "ShallowDiagnosticAgent":
        """
        Train the organizational agent.
        """

        if len(X) == 0:
            raise ValueError(
                f"{self.config.agent_name}: empty training matrix."
            )

        if len(X) != len(y):
            raise ValueError(
                f"{self.config.agent_name}: inconsistent X/y lengths."
            )

        self.model.fit(X, y)

        self.is_fitted = True

        train_prob = self.predict_proba(X)
        train_pred = (
            train_prob >= self.config.probability_threshold
        ).astype(int)

        self.training_metrics_ = self._compute_metrics(
            np.asarray(y),
            train_pred,
            train_prob,
        )

        self.feature_importance_ = self._extract_feature_importance()

        return self

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """
        Binary predictions.
        """

        self._ensure_fitted()

        probs = self.predict_proba(X)

        return (
            probs >= self.config.probability_threshold
        ).astype(int)

    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """
        Positive-class probabilities.
        """

        self._ensure_fitted()

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]

        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)

            scores = (
                scores - np.min(scores)
            ) / (
                np.max(scores)
                - np.min(scores)
                + 1e-8
            )

            return scores

        return self.model.predict(X).astype(float)

    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate agent performance.
        """

        self._ensure_fitted()

        probabilities = self.predict_proba(X)

        predictions = (
            probabilities
            >= self.config.probability_threshold
        ).astype(int)

        metrics = self._compute_metrics(
            np.asarray(y),
            predictions,
            probabilities,
        )

        metrics["agent_name"] = self.config.agent_name
        metrics["agent_type"] = self.config.agent_type

        return metrics

    def export_predictions(
        self,
        X: pd.DataFrame | np.ndarray,
        record_ids: Optional[List[Any]] = None,
    ) -> pd.DataFrame:
        """
        Export agent predictions for coordination/fusion stages.
        """

        self._ensure_fitted()

        probabilities = self.predict_proba(X)
        predictions = self.predict(X)

        output = pd.DataFrame(
            {
                "probability": probabilities,
                "prediction": predictions,
            }
        )

        if record_ids is not None:
            output.insert(
                0,
                "record_id",
                record_ids,
            )

        output.insert(
            0,
            "agent_name",
            self.config.agent_name,
        )

        return output

    def summary(self) -> pd.DataFrame:
        """
        Organizational summary of agent state.
        """

        rows = [
            {
                "attribute": "agent_name",
                "value": self.config.agent_name,
            },
            {
                "attribute": "agent_type",
                "value": self.config.agent_type,
            },
            {
                "attribute": "is_fitted",
                "value": self.is_fitted,
            },
        ]

        for metric, value in self.training_metrics_.items():
            rows.append(
                {
                    "attribute": metric,
                    "value": value,
                }
            )

        return pd.DataFrame(rows)

    def _ensure_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.config.agent_name} has not been trained."
            )

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, float]:

        results = {
            "accuracy": accuracy_score(
                y_true,
                y_pred,
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_true,
                y_pred,
            ),
            "precision": precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "f1": f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "mcc": matthews_corrcoef(
                y_true,
                y_pred,
            ),
        }

        try:
            results["roc_auc"] = roc_auc_score(
                y_true,
                y_prob,
            )
        except Exception:
            results["roc_auc"] = np.nan

        try:
            results["average_precision"] = (
                average_precision_score(
                    y_true,
                    y_prob,
                )
            )
        except Exception:
            results["average_precision"] = np.nan

        return results

    def _extract_feature_importance(
        self,
    ) -> Optional[pd.DataFrame]:
        """
        Extract pathway feature importance.
        """

        estimator = self.model

        if isinstance(estimator, Pipeline):
            estimator = estimator.named_steps["classifier"]

        feature_names = self.config.feature_names

        if not feature_names:
            return None

        rows = []

        if hasattr(estimator, "feature_importances_"):

            for feature, importance in zip(
                feature_names,
                estimator.feature_importances_,
            ):
                rows.append(
                    {
                        "agent_name": self.config.agent_name,
                        "feature": feature,
                        "importance": float(importance),
                    }
                )

        elif hasattr(estimator, "coef_"):

            coefficients = np.abs(
                estimator.coef_.ravel()
            )

            for feature, coef in zip(
                feature_names,
                coefficients,
            ):
                rows.append(
                    {
                        "agent_name": self.config.agent_name,
                        "feature": feature,
                        "importance": float(coef),
                    }
                )

        if not rows:
            return None

        return (
            pd.DataFrame(rows)
            .sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )


def clone_agent(
    agent: ShallowDiagnosticAgent,
) -> ShallowDiagnosticAgent:
    """
    Create an unfitted clone of an existing agent.
    """

    return ShallowDiagnosticAgent(
        config=agent.config,
    )


def summarize_agent_collection(
    agents: Dict[str, ShallowDiagnosticAgent],
) -> pd.DataFrame:
    """
    Summarize a collection of SACU agents.
    """

    rows = []

    for name, agent in agents.items():

        row = {
            "agent_name": name,
            "agent_type": agent.config.agent_type,
            "is_fitted": agent.is_fitted,
        }

        row.update(agent.training_metrics_)

        rows.append(row)

    return pd.DataFrame(rows)
