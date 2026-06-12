# src/organizational_sacu_mammography/evaluation/ablation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import compute_classification_metrics


@dataclass
class AblationConfig:
    """
    Configuration for organizational pathway ablation analysis.

    This module evaluates the effect of removing one SACU pathway at a time
    from coordinated or fused organizational inference.
    """

    threshold: float = 0.50
    probability_suffix: str = "_probability"
    weight_suffix: str = "_weight"
    eps: float = 1e-8


def _validate_probability_frame(
    probability_df: pd.DataFrame,
) -> None:
    if probability_df.empty:
        raise ValueError("probability_df is empty.")

    if probability_df.isna().any().any():
        raise ValueError("probability_df contains NaN values.")

    if np.isinf(probability_df.to_numpy()).any():
        raise ValueError("probability_df contains infinite values.")


def _agent_name_from_probability_column(
    column: str,
    config: AblationConfig,
) -> str:
    if column.endswith(config.probability_suffix):
        return column[: -len(config.probability_suffix)]
    return column


def _weight_column_for_agent(
    agent_name: str,
    config: AblationConfig,
) -> str:
    return f"{agent_name}{config.weight_suffix}"


def compute_weighted_probability(
    probability_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    config: AblationConfig = AblationConfig(),
) -> pd.Series:
    """
    Compute weighted SACU probability from agent probabilities and weights.
    """

    _validate_probability_frame(probability_df)

    if weight_df.empty:
        raise ValueError("weight_df is empty.")

    if len(probability_df) != len(weight_df):
        raise ValueError("probability_df and weight_df have inconsistent lengths.")

    weighted_terms = []

    for probability_col in probability_df.columns:
        agent_name = _agent_name_from_probability_column(
            probability_col,
            config,
        )

        weight_col = _weight_column_for_agent(
            agent_name,
            config,
        )

        if weight_col not in weight_df.columns:
            raise KeyError(f"Missing weight column: {weight_col}")

        weighted_terms.append(
            probability_df[probability_col].to_numpy(dtype=float)
            * weight_df[weight_col].to_numpy(dtype=float)
        )

    probability = np.sum(
        np.vstack(weighted_terms),
        axis=0,
    )

    return pd.Series(
        probability,
        index=probability_df.index,
        name="weighted_sacu_probability",
    )


def remove_agent_and_renormalize(
    probability_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    agent_name: str,
    config: AblationConfig = AblationConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove one pathway and renormalize the remaining SACU weights.
    """

    probability_col = f"{agent_name}{config.probability_suffix}"
    weight_col = f"{agent_name}{config.weight_suffix}"

    if probability_col not in probability_df.columns:
        raise KeyError(f"Missing probability column: {probability_col}")

    if weight_col not in weight_df.columns:
        raise KeyError(f"Missing weight column: {weight_col}")

    reduced_probabilities = probability_df.drop(
        columns=[probability_col],
    )

    reduced_weights = weight_df.drop(
        columns=[weight_col],
    )

    row_sum = reduced_weights.sum(axis=1)

    zero_rows = row_sum <= config.eps

    if zero_rows.any():
        reduced_weights.loc[zero_rows, :] = 1.0
        row_sum = reduced_weights.sum(axis=1)

    reduced_weights = reduced_weights.div(
        row_sum + config.eps,
        axis=0,
    )

    return reduced_probabilities, reduced_weights


def run_pathway_ablation(
    y_true: np.ndarray | pd.Series,
    probability_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    reference_probability: Optional[pd.Series] = None,
    config: AblationConfig = AblationConfig(),
) -> pd.DataFrame:
    """
    Run leave-one-pathway-out organizational ablation.

    If reference_probability is not supplied, the full weighted SACU
    probability is computed from probability_df and weight_df.
    """

    _validate_probability_frame(probability_df)

    y = np.asarray(y_true).astype(int)

    if len(y) != len(probability_df):
        raise ValueError("y_true and probability_df have inconsistent lengths.")

    if reference_probability is None:
        reference_probability = compute_weighted_probability(
            probability_df,
            weight_df,
            config,
        )

    full_metrics = compute_classification_metrics(
        y_true=y,
        y_probability=reference_probability,
        threshold=config.threshold,
        model_name="FullSACU",
    )

    full_metrics["ablation_type"] = "full_model"
    full_metrics["removed_agent"] = "none"

    rows: List[Dict[str, float | str]] = [full_metrics]

    for probability_col in probability_df.columns:
        agent_name = _agent_name_from_probability_column(
            probability_col,
            config,
        )

        reduced_probabilities, reduced_weights = remove_agent_and_renormalize(
            probability_df=probability_df,
            weight_df=weight_df,
            agent_name=agent_name,
            config=config,
        )

        ablated_probability = compute_weighted_probability(
            reduced_probabilities,
            reduced_weights,
            config=config,
        )

        metrics = compute_classification_metrics(
            y_true=y,
            y_probability=ablated_probability,
            threshold=config.threshold,
            model_name=f"Without_{agent_name}",
        )

        metrics["ablation_type"] = "leave_one_pathway_out"
        metrics["removed_agent"] = agent_name

        for metric_name in [
            "roc_auc",
            "average_precision",
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1",
            "mcc",
        ]:
            metrics[f"delta_{metric_name}"] = (
                full_metrics[metric_name] - metrics[metric_name]
            )

        rows.append(metrics)

    return pd.DataFrame(rows)


def compute_role_contribution_loss(
    ablation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert ablation results into pathway contribution-loss table.
    """

    if ablation_df.empty:
        return pd.DataFrame()

    ablated = ablation_df[
        ablation_df["ablation_type"] == "leave_one_pathway_out"
    ].copy()

    if ablated.empty:
        return pd.DataFrame()

    columns = [
        "removed_agent",
        "delta_roc_auc",
        "delta_average_precision",
        "delta_balanced_accuracy",
        "delta_sensitivity",
        "delta_specificity",
        "delta_f1",
        "delta_mcc",
    ]

    available = [c for c in columns if c in ablated.columns]

    out = ablated[available].copy()

    return (
        out.sort_values(
            "delta_roc_auc",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def summarize_ablation_results(
    ablation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize organizational ablation findings.
    """

    if ablation_df.empty:
        return pd.DataFrame(
            columns=["metric", "value"]
        )

    ablated = ablation_df[
        ablation_df["ablation_type"] == "leave_one_pathway_out"
    ].copy()

    if ablated.empty:
        return pd.DataFrame()

    largest_auc_loss = ablated.sort_values(
        "delta_roc_auc",
        ascending=False,
    ).iloc[0]

    largest_balanced_loss = ablated.sort_values(
        "delta_balanced_accuracy",
        ascending=False,
    ).iloc[0]

    return pd.DataFrame(
        [
            {
                "metric": "n_ablation_conditions",
                "value": int(len(ablated)),
            },
            {
                "metric": "largest_roc_auc_loss_agent",
                "value": largest_auc_loss["removed_agent"],
            },
            {
                "metric": "largest_roc_auc_loss",
                "value": float(largest_auc_loss["delta_roc_auc"]),
            },
            {
                "metric": "largest_balanced_accuracy_loss_agent",
                "value": largest_balanced_loss["removed_agent"],
            },
            {
                "metric": "largest_balanced_accuracy_loss",
                "value": float(largest_balanced_loss["delta_balanced_accuracy"]),
            },
        ]
    )


def build_ablation_audit_table(
    ablation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build compact manuscript-ready ablation audit table.
    """

    if ablation_df.empty:
        return pd.DataFrame()

    cols = [
        "model",
        "removed_agent",
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "delta_roc_auc",
        "delta_balanced_accuracy",
        "delta_f1",
    ]

    available_cols = [c for c in cols if c in ablation_df.columns]

    return ablation_df[available_cols].copy()
