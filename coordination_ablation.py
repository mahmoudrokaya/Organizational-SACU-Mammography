# src/organizational_sacu_mammography/evaluation/coordination_ablation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import compute_classification_metrics


@dataclass
class CoordinationAblationConfig:
    """
    Configuration for SACU coordination-mechanism ablation.

    This module evaluates whether adaptive SACU coordination improves over:
    - uniform averaging,
    - reliability-only weighting,
    - confidence-only weighting,
    - static mean-weight coordination,
    - resource-limited top-k activation.
    """

    threshold: float = 0.50
    probability_suffix: str = "_probability"
    weight_suffix: str = "_weight"
    eps: float = 1e-8
    top_k_values: tuple[int, ...] = (1, 2, 3)


def _validate_probability_frame(
    probability_df: pd.DataFrame,
) -> None:
    if probability_df.empty:
        raise ValueError("probability_df is empty.")

    if probability_df.isna().any().any():
        raise ValueError("probability_df contains NaN values.")

    if np.isinf(probability_df.to_numpy()).any():
        raise ValueError("probability_df contains infinite values.")


def _validate_weight_frame(
    weight_df: pd.DataFrame,
    probability_df: pd.DataFrame,
) -> None:
    if weight_df.empty:
        raise ValueError("weight_df is empty.")

    if len(weight_df) != len(probability_df):
        raise ValueError("weight_df and probability_df have inconsistent lengths.")

    if weight_df.isna().any().any():
        raise ValueError("weight_df contains NaN values.")

    if np.isinf(weight_df.to_numpy()).any():
        raise ValueError("weight_df contains infinite values.")


def _weighted_probability(
    probability_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    eps: float = 1e-8,
) -> pd.Series:
    row_sum = weight_df.sum(axis=1)

    normalized = weight_df.div(
        row_sum + eps,
        axis=0,
    )

    probability = np.sum(
        probability_df.to_numpy(dtype=float)
        * normalized.to_numpy(dtype=float),
        axis=1,
    )

    return pd.Series(
        probability,
        index=probability_df.index,
        name="coordinated_probability",
    )


def uniform_coordination(
    probability_df: pd.DataFrame,
) -> pd.Series:
    """
    Compute unweighted mean SACU probability.
    """

    _validate_probability_frame(probability_df)

    return pd.Series(
        probability_df.mean(axis=1),
        index=probability_df.index,
        name="uniform_probability",
    )


def static_mean_weight_coordination(
    probability_df: pd.DataFrame,
    adaptive_weights: pd.DataFrame,
    config: CoordinationAblationConfig = CoordinationAblationConfig(),
) -> pd.Series:
    """
    Replace case-specific adaptive weights with cohort-level mean weights.
    """

    _validate_probability_frame(probability_df)
    _validate_weight_frame(adaptive_weights, probability_df)

    mean_weights = adaptive_weights.mean(axis=0)

    static_weights = pd.DataFrame(
        np.tile(
            mean_weights.to_numpy(dtype=float),
            (len(probability_df), 1),
        ),
        index=probability_df.index,
        columns=adaptive_weights.columns,
    )

    return _weighted_probability(
        probability_df,
        static_weights,
        eps=config.eps,
    ).rename("static_mean_weight_probability")


def confidence_only_coordination(
    probability_df: pd.DataFrame,
    config: CoordinationAblationConfig = CoordinationAblationConfig(),
) -> pd.Series:
    """
    Use prediction confidence as the only coordination signal.
    """

    _validate_probability_frame(probability_df)

    confidence = (probability_df - 0.5).abs() * 2.0

    confidence = confidence + config.eps

    return _weighted_probability(
        probability_df,
        confidence,
        eps=config.eps,
    ).rename("confidence_only_probability")


def reliability_only_coordination(
    probability_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    config: CoordinationAblationConfig = CoordinationAblationConfig(),
) -> pd.Series:
    """
    Use validation reliability as a static coordination signal.
    """

    _validate_probability_frame(probability_df)

    if reliability_df.empty:
        raise ValueError("reliability_df is empty.")

    if "agent_name" not in reliability_df.columns:
        raise KeyError("reliability_df must contain agent_name.")

    if "reliability_weight" not in reliability_df.columns:
        raise KeyError("reliability_df must contain reliability_weight.")

    weights = []

    for probability_col in probability_df.columns:
        agent_name = probability_col.replace(
            config.probability_suffix,
            "",
        )

        row = reliability_df[
            reliability_df["agent_name"] == agent_name
        ]

        if row.empty:
            weights.append(1.0)
        else:
            weights.append(float(row.iloc[0]["reliability_weight"]))

    static_weights = pd.DataFrame(
        np.tile(
            np.asarray(weights, dtype=float),
            (len(probability_df), 1),
        ),
        index=probability_df.index,
        columns=probability_df.columns,
    )

    return _weighted_probability(
        probability_df,
        static_weights,
        eps=config.eps,
    ).rename("reliability_only_probability")


def adaptive_top_k_coordination(
    probability_df: pd.DataFrame,
    adaptive_weights: pd.DataFrame,
    k: int,
    config: CoordinationAblationConfig = CoordinationAblationConfig(),
) -> pd.Series:
    """
    Keep only the top-k highest-weight agents per case.
    """

    _validate_probability_frame(probability_df)
    _validate_weight_frame(adaptive_weights, probability_df)

    if k <= 0:
        raise ValueError("k must be positive.")

    if k > adaptive_weights.shape[1]:
        raise ValueError("k cannot exceed the number of agents.")

    reduced_weights = pd.DataFrame(
        0.0,
        index=adaptive_weights.index,
        columns=adaptive_weights.columns,
    )

    values = adaptive_weights.to_numpy(dtype=float)

    top_indices = np.argsort(values, axis=1)[:, -k:]

    for row_idx in range(values.shape[0]):
        reduced_weights.iloc[
            row_idx,
            top_indices[row_idx],
        ] = adaptive_weights.iloc[
            row_idx,
            top_indices[row_idx],
        ]

    return _weighted_probability(
        probability_df,
        reduced_weights,
        eps=config.eps,
    ).rename(f"adaptive_top_{k}_probability")


def evaluate_coordination_mechanisms(
    y_true: np.ndarray | pd.Series,
    probability_df: pd.DataFrame,
    adaptive_weights: pd.DataFrame,
    reliability_df: Optional[pd.DataFrame] = None,
    reference_probability: Optional[pd.Series] = None,
    config: CoordinationAblationConfig = CoordinationAblationConfig(),
) -> Dict[str, pd.DataFrame]:
    """
    Evaluate SACU coordination mechanisms and ablations.
    """

    _validate_probability_frame(probability_df)
    _validate_weight_frame(adaptive_weights, probability_df)

    y = np.asarray(y_true).astype(int)

    if len(y) != len(probability_df):
        raise ValueError("y_true and probability_df have inconsistent lengths.")

    mechanism_probabilities: Dict[str, pd.Series] = {}

    if reference_probability is not None:
        mechanism_probabilities["AdaptiveCoordination"] = reference_probability
    else:
        mechanism_probabilities["AdaptiveCoordination"] = _weighted_probability(
            probability_df,
            adaptive_weights,
            eps=config.eps,
        ).rename("adaptive_probability")

    mechanism_probabilities["UniformMean"] = uniform_coordination(
        probability_df,
    )

    mechanism_probabilities["StaticMeanWeight"] = static_mean_weight_coordination(
        probability_df,
        adaptive_weights,
        config,
    )

    mechanism_probabilities["ConfidenceOnly"] = confidence_only_coordination(
        probability_df,
        config,
    )

    if reliability_df is not None and not reliability_df.empty:
        mechanism_probabilities["ReliabilityOnly"] = reliability_only_coordination(
            probability_df,
            reliability_df,
            config,
        )

    for k in config.top_k_values:
        if k <= probability_df.shape[1]:
            mechanism_probabilities[f"AdaptiveTop{k}"] = adaptive_top_k_coordination(
                probability_df,
                adaptive_weights,
                k,
                config,
            )

    performance_rows: List[Dict[str, float | str]] = []

    for mechanism_name, probabilities in mechanism_probabilities.items():
        metrics = compute_classification_metrics(
            y_true=y,
            y_probability=probabilities,
            threshold=config.threshold,
            model_name=mechanism_name,
        )

        metrics["coordination_mechanism"] = mechanism_name
        performance_rows.append(metrics)

    performance_df = pd.DataFrame(performance_rows)

    paired_df = build_coordination_paired_comparison(
        performance_df,
        reference_mechanism="AdaptiveCoordination",
    )

    variability_df = compute_case_level_adaptive_variability(
        probability_df=probability_df,
        adaptive_weights=adaptive_weights,
        config=config,
    )

    topk_df = performance_df[
        performance_df["coordination_mechanism"]
        .astype(str)
        .str.startswith("AdaptiveTop")
    ].copy()

    entropy_df = summarize_adaptive_weight_entropy(
        adaptive_weights,
        config=config,
    )

    evidence_df = build_coordination_response_evidence_table(
        performance_df=performance_df,
        paired_df=paired_df,
        entropy_df=entropy_df,
    )

    return {
        "coordination_ablation_performance": performance_df,
        "coordination_paired_comparison": paired_df,
        "case_level_adaptive_variability": variability_df,
        "resource_limited_topk_performance": topk_df,
        "adaptive_weight_entropy_summary": entropy_df,
        "coordination_response_evidence_table": evidence_df,
    }


def build_coordination_paired_comparison(
    performance_df: pd.DataFrame,
    reference_mechanism: str = "AdaptiveCoordination",
) -> pd.DataFrame:
    """
    Compare all coordination mechanisms against the adaptive reference.
    """

    if performance_df.empty:
        return pd.DataFrame()

    reference = performance_df[
        performance_df["coordination_mechanism"] == reference_mechanism
    ]

    if reference.empty:
        raise ValueError(f"Reference mechanism not found: {reference_mechanism}")

    reference_row = reference.iloc[0]

    rows = []

    for _, row in performance_df.iterrows():
        mechanism = row["coordination_mechanism"]

        if mechanism == reference_mechanism:
            continue

        rows.append(
            {
                "reference_mechanism": reference_mechanism,
                "comparison_mechanism": mechanism,
                "delta_roc_auc": float(reference_row["roc_auc"] - row["roc_auc"]),
                "delta_average_precision": float(
                    reference_row["average_precision"] - row["average_precision"]
                ),
                "delta_balanced_accuracy": float(
                    reference_row["balanced_accuracy"] - row["balanced_accuracy"]
                ),
                "delta_sensitivity": float(
                    reference_row["sensitivity"] - row["sensitivity"]
                ),
                "delta_specificity": float(
                    reference_row["specificity"] - row["specificity"]
                ),
                "delta_f1": float(reference_row["f1"] - row["f1"]),
                "delta_mcc": float(reference_row["mcc"] - row["mcc"]),
            }
        )

    return pd.DataFrame(rows)


def compute_case_level_adaptive_variability(
    probability_df: pd.DataFrame,
    adaptive_weights: pd.DataFrame,
    config: CoordinationAblationConfig = CoordinationAblationConfig(),
) -> pd.DataFrame:
    """
    Quantify case-level variability in adaptive SACU coordination.
    """

    _validate_probability_frame(probability_df)
    _validate_weight_frame(adaptive_weights, probability_df)

    weighted_probability = _weighted_probability(
        probability_df,
        adaptive_weights,
        eps=config.eps,
    )

    uniform_probability = uniform_coordination(
        probability_df,
    )

    dominant_weight_col = adaptive_weights.idxmax(axis=1)

    return pd.DataFrame(
        {
            "adaptive_probability": weighted_probability,
            "uniform_probability": uniform_probability,
            "adaptive_minus_uniform": weighted_probability - uniform_probability,
            "dominant_weight": adaptive_weights.max(axis=1),
            "dominant_agent": dominant_weight_col.str.replace(
                config.weight_suffix,
                "",
                regex=False,
            ),
            "weight_range": adaptive_weights.max(axis=1) - adaptive_weights.min(axis=1),
            "weight_std": adaptive_weights.std(axis=1),
        },
        index=probability_df.index,
    )


def summarize_adaptive_weight_entropy(
    adaptive_weights: pd.DataFrame,
    config: CoordinationAblationConfig = CoordinationAblationConfig(),
) -> pd.DataFrame:
    """
    Summarize entropy of adaptive influence weights.
    """

    if adaptive_weights.empty:
        return pd.DataFrame()

    weights = adaptive_weights.to_numpy(dtype=float)

    entropy = -np.sum(
        weights * np.log(weights + config.eps),
        axis=1,
    )

    max_entropy = np.log(adaptive_weights.shape[1])

    normalized_entropy = entropy / (max_entropy + config.eps)

    return pd.DataFrame(
        [
            {
                "n_cases": int(len(adaptive_weights)),
                "n_agents": int(adaptive_weights.shape[1]),
                "mean_entropy": float(np.mean(entropy)),
                "std_entropy": float(np.std(entropy)),
                "min_entropy": float(np.min(entropy)),
                "max_entropy": float(np.max(entropy)),
                "mean_normalized_entropy": float(np.mean(normalized_entropy)),
                "std_normalized_entropy": float(np.std(normalized_entropy)),
            }
        ]
    )


def build_coordination_response_evidence_table(
    performance_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    entropy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build compact reviewer-response evidence table.
    """

    rows = []

    if not performance_df.empty:
        adaptive = performance_df[
            performance_df["coordination_mechanism"] == "AdaptiveCoordination"
        ]

        if not adaptive.empty:
            row = adaptive.iloc[0]

            rows.append(
                {
                    "evidence_item": "Adaptive coordination performance",
                    "metric": "ROC-AUC",
                    "value": float(row["roc_auc"]),
                }
            )

            rows.append(
                {
                    "evidence_item": "Adaptive coordination performance",
                    "metric": "Balanced accuracy",
                    "value": float(row["balanced_accuracy"]),
                }
            )

    if not paired_df.empty:
        best_delta = paired_df.sort_values(
            "delta_roc_auc",
            ascending=False,
        ).iloc[0]

        rows.append(
            {
                "evidence_item": "Largest ROC-AUC advantage over alternative coordination",
                "metric": best_delta["comparison_mechanism"],
                "value": float(best_delta["delta_roc_auc"]),
            }
        )

    if not entropy_df.empty:
        row = entropy_df.iloc[0]

        rows.append(
            {
                "evidence_item": "Case-level adaptive-weight diversity",
                "metric": "mean_normalized_entropy",
                "value": float(row["mean_normalized_entropy"]),
            }
        )

    return pd.DataFrame(rows)


def summarize_coordination_ablation(
    outputs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Summarize coordination-ablation outputs.
    """

    performance_df = outputs.get(
        "coordination_ablation_performance",
        pd.DataFrame(),
    )

    paired_df = outputs.get(
        "coordination_paired_comparison",
        pd.DataFrame(),
    )

    entropy_df = outputs.get(
        "adaptive_weight_entropy_summary",
        pd.DataFrame(),
    )

    rows = []

    if not performance_df.empty:
        adaptive = performance_df[
            performance_df["coordination_mechanism"] == "AdaptiveCoordination"
        ]

        if not adaptive.empty:
            row = adaptive.iloc[0]

            rows.append(
                {
                    "metric": "adaptive_coordination_roc_auc",
                    "value": float(row["roc_auc"]),
                }
            )

            rows.append(
                {
                    "metric": "adaptive_coordination_balanced_accuracy",
                    "value": float(row["balanced_accuracy"]),
                }
            )

    if not paired_df.empty:
        strongest = paired_df.sort_values(
            "delta_roc_auc",
            ascending=False,
        ).iloc[0]

        rows.append(
            {
                "metric": "strongest_comparison_advantage",
                "value": strongest["comparison_mechanism"],
            }
        )

        rows.append(
            {
                "metric": "strongest_delta_roc_auc",
                "value": float(strongest["delta_roc_auc"]),
            }
        )

    if not entropy_df.empty:
        row = entropy_df.iloc[0]

        rows.append(
            {
                "metric": "mean_normalized_weight_entropy",
                "value": float(row["mean_normalized_entropy"]),
            }
        )

    return pd.DataFrame(rows)
