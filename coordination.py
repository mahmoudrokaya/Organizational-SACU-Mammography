# src/organizational_sacu_mammography/models/coordination.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score


@dataclass
class CoordinationConfig:
    """
    Configuration for SACU organizational coordination.

    The coordination layer converts agent outputs, validation reliability,
    and case-level pathway availability into adaptive influence weights.
    """

    eps: float = 1e-8
    reliability_metric: str = "roc_auc"

    use_reliability_weighting: bool = True
    use_case_availability_weighting: bool = True
    use_confidence_weighting: bool = True

    minimum_weight: float = 0.0
    normalize_weights: bool = True

    probability_suffix: str = "_probability"
    mask_suffix: str = "_mask"


def _validate_probability_frame(
    agent_probabilities: pd.DataFrame,
) -> None:
    if agent_probabilities.empty:
        raise ValueError("agent_probabilities is empty.")

    if agent_probabilities.isna().any().any():
        raise ValueError("agent_probabilities contains NaN values.")

    if np.isinf(agent_probabilities.to_numpy()).any():
        raise ValueError("agent_probabilities contains infinite values.")


def _agent_name_from_probability_column(
    column: str,
    suffix: str,
) -> str:
    if column.endswith(suffix):
        return column[: -len(suffix)]

    return column


def _mask_name_for_agent(
    agent_name: str,
    config: CoordinationConfig,
) -> str:
    normalized = (
        agent_name.replace("Agent", "")
        .replace("LocalRegional", "local_regional")
        .replace("MultiView", "multiview")
        .replace("Bilateral", "bilateral")
        .replace("TemporalSpatial", "temporal_spatial")
        .replace("Metadata", "metadata")
        .replace("AdaptiveControl", "adaptive_control")
    )

    return f"{normalized}{config.mask_suffix}"


def compute_agent_reliability(
    validation_probabilities: pd.DataFrame,
    y_validation: pd.Series | np.ndarray,
    config: CoordinationConfig = CoordinationConfig(),
) -> pd.DataFrame:
    """
    Estimate validation reliability for each SACU agent.

    Reliability is used as the global organizational competence prior.
    """

    _validate_probability_frame(validation_probabilities)

    y = np.asarray(y_validation)

    if len(validation_probabilities) != len(y):
        raise ValueError(
            "validation_probabilities and y_validation have inconsistent lengths."
        )

    rows: List[Dict[str, float | str]] = []

    for column in validation_probabilities.columns:
        probs = validation_probabilities[column].to_numpy(dtype=float)
        preds = (probs >= 0.5).astype(int)

        agent_name = _agent_name_from_probability_column(
            column,
            config.probability_suffix,
        )

        try:
            roc_auc = roc_auc_score(y, probs)
        except Exception:
            roc_auc = np.nan

        try:
            balanced_acc = balanced_accuracy_score(y, preds)
        except Exception:
            balanced_acc = np.nan

        try:
            f1 = f1_score(y, preds, zero_division=0)
        except Exception:
            f1 = np.nan

        metric_map = {
            "roc_auc": roc_auc,
            "balanced_accuracy": balanced_acc,
            "f1": f1,
        }

        selected = metric_map.get(
            config.reliability_metric,
            roc_auc,
        )

        if np.isnan(selected):
            selected = 0.5

        rows.append(
            {
                "agent_name": agent_name,
                "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else np.nan,
                "balanced_accuracy": (
                    float(balanced_acc) if not np.isnan(balanced_acc) else np.nan
                ),
                "f1": float(f1) if not np.isnan(f1) else np.nan,
                "reliability": float(max(selected, config.eps)),
            }
        )

    reliability_df = pd.DataFrame(rows)

    total = reliability_df["reliability"].sum()

    reliability_df["reliability_weight"] = (
        reliability_df["reliability"] / (total + config.eps)
    )

    return reliability_df


def _confidence_from_probabilities(
    probabilities: pd.Series,
) -> pd.Series:
    """
    Convert probabilities to confidence scores.

    Confidence is high when probability is far from 0.5.
    """

    return (probabilities - 0.5).abs() * 2.0


def compute_adaptive_agent_weights(
    agent_probabilities: pd.DataFrame,
    reliability_df: Optional[pd.DataFrame] = None,
    pathway_masks: Optional[pd.DataFrame] = None,
    config: CoordinationConfig = CoordinationConfig(),
) -> pd.DataFrame:
    """
    Compute case-level adaptive SACU influence weights.

    The final weight combines:
    - global validation reliability,
    - case-level pathway availability,
    - prediction confidence.
    """

    _validate_probability_frame(agent_probabilities)

    weights = pd.DataFrame(
        index=agent_probabilities.index,
    )

    reliability_map: Dict[str, float] = {}

    if reliability_df is not None and not reliability_df.empty:
        if "agent_name" not in reliability_df.columns:
            raise KeyError("reliability_df must contain agent_name.")

        if "reliability_weight" not in reliability_df.columns:
            raise KeyError("reliability_df must contain reliability_weight.")

        reliability_map = dict(
            zip(
                reliability_df["agent_name"],
                reliability_df["reliability_weight"],
            )
        )

    for column in agent_probabilities.columns:
        agent_name = _agent_name_from_probability_column(
            column,
            config.probability_suffix,
        )

        base = pd.Series(
            1.0,
            index=agent_probabilities.index,
            dtype=float,
        )

        if config.use_reliability_weighting:
            base *= reliability_map.get(
                agent_name,
                1.0,
            )

        if config.use_case_availability_weighting and pathway_masks is not None:
            mask_col = _mask_name_for_agent(
                agent_name,
                config,
            )

            if mask_col in pathway_masks.columns:
                base *= pathway_masks.loc[
                    agent_probabilities.index,
                    mask_col,
                ].fillna(0).astype(float)

        if config.use_confidence_weighting:
            confidence = _confidence_from_probabilities(
                agent_probabilities[column]
            )
            base *= confidence + config.eps

        base = base.clip(lower=config.minimum_weight)

        weights[f"{agent_name}_weight"] = base

    if config.normalize_weights:
        row_sum = weights.sum(axis=1)

        zero_rows = row_sum <= config.eps

        if zero_rows.any():
            weights.loc[zero_rows, :] = 1.0
            row_sum = weights.sum(axis=1)

        weights = weights.div(
            row_sum + config.eps,
            axis=0,
        )

    return weights


def apply_coordination_weights(
    agent_probabilities: pd.DataFrame,
    adaptive_weights: pd.DataFrame,
    config: CoordinationConfig = CoordinationConfig(),
) -> pd.Series:
    """
    Compute coordinated SACU probability from weighted agent outputs.
    """

    _validate_probability_frame(agent_probabilities)

    if len(agent_probabilities) != len(adaptive_weights):
        raise ValueError(
            "agent_probabilities and adaptive_weights have inconsistent lengths."
        )

    aligned_terms = []

    for probability_col in agent_probabilities.columns:
        agent_name = _agent_name_from_probability_column(
            probability_col,
            config.probability_suffix,
        )

        weight_col = f"{agent_name}_weight"

        if weight_col not in adaptive_weights.columns:
            raise KeyError(f"Missing adaptive weight column: {weight_col}")

        aligned_terms.append(
            agent_probabilities[probability_col].to_numpy(dtype=float)
            * adaptive_weights[weight_col].to_numpy(dtype=float)
        )

    combined = np.sum(
        np.vstack(aligned_terms),
        axis=0,
    )

    return pd.Series(
        combined,
        index=agent_probabilities.index,
        name="coordinated_probability",
    )


def summarize_coordination_weights(
    adaptive_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize adaptive organizational influence weights.
    """

    if adaptive_weights.empty:
        return pd.DataFrame(
            columns=[
                "agent",
                "mean_weight",
                "std_weight",
                "min_weight",
                "max_weight",
            ]
        )

    rows = []

    for col in adaptive_weights.columns:
        values = adaptive_weights[col].astype(float)

        rows.append(
            {
                "agent": col.replace("_weight", ""),
                "mean_weight": float(values.mean()),
                "std_weight": float(values.std()),
                "min_weight": float(values.min()),
                "max_weight": float(values.max()),
                "dominant_cases": int(
                    (adaptive_weights.idxmax(axis=1) == col).sum()
                ),
                "dominant_fraction": float(
                    (adaptive_weights.idxmax(axis=1) == col).mean()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "mean_weight",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def compute_weight_entropy(
    adaptive_weights: pd.DataFrame,
    eps: float = 1e-8,
) -> pd.Series:
    """
    Compute case-level entropy of adaptive SACU weights.
    """

    weights = adaptive_weights.to_numpy(dtype=float)
    entropy = -np.sum(
        weights * np.log(weights + eps),
        axis=1,
    )

    return pd.Series(
        entropy,
        index=adaptive_weights.index,
        name="adaptive_weight_entropy",
    )


def audit_coordination_outputs(
    agent_probabilities: pd.DataFrame,
    adaptive_weights: pd.DataFrame,
    coordinated_probability: pd.Series,
) -> pd.DataFrame:
    """
    Build case-level audit table for SACU coordination.
    """

    if len(agent_probabilities) != len(adaptive_weights):
        raise ValueError("Probability and weight frames have inconsistent lengths.")

    audit = pd.DataFrame(index=agent_probabilities.index)

    audit["coordinated_probability"] = coordinated_probability
    audit["coordinated_prediction"] = (
        coordinated_probability >= 0.5
    ).astype(int)

    audit["dominant_agent"] = adaptive_weights.idxmax(axis=1).str.replace(
        "_weight",
        "",
        regex=False,
    )

    audit["dominant_weight"] = adaptive_weights.max(axis=1)
    audit["adaptive_weight_entropy"] = compute_weight_entropy(
        adaptive_weights,
    )

    return pd.concat(
        [
            audit,
            agent_probabilities.add_prefix("prob_"),
            adaptive_weights.add_prefix("weight_"),
        ],
        axis=1,
    )
