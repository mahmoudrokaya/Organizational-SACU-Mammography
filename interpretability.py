# src/organizational_sacu_mammography/evaluation/interpretability.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import compute_classification_metrics


@dataclass
class InterpretabilityConfig:
    """
    Configuration for organizational SACU interpretability analysis.

    This module quantifies pathway-level contribution using:
    - adaptive influence weights,
    - dominant-pathway frequency,
    - individual agent performance,
    - ablation-based contribution loss.
    """

    threshold: float = 0.50
    probability_suffix: str = "_probability"
    weight_suffix: str = "_weight"
    eps: float = 1e-8


def _agent_name_from_probability_column(
    column: str,
    config: InterpretabilityConfig,
) -> str:
    if column.endswith(config.probability_suffix):
        return column[: -len(config.probability_suffix)]
    return column


def _agent_name_from_weight_column(
    column: str,
    config: InterpretabilityConfig,
) -> str:
    if column.endswith(config.weight_suffix):
        return column[: -len(config.weight_suffix)]
    return column


def _validate_frames(
    agent_probabilities: pd.DataFrame,
    adaptive_weights: pd.DataFrame,
) -> None:
    if agent_probabilities.empty:
        raise ValueError("agent_probabilities is empty.")

    if adaptive_weights.empty:
        raise ValueError("adaptive_weights is empty.")

    if len(agent_probabilities) != len(adaptive_weights):
        raise ValueError(
            "agent_probabilities and adaptive_weights have inconsistent lengths."
        )

    if agent_probabilities.isna().any().any():
        raise ValueError("agent_probabilities contains NaN values.")

    if adaptive_weights.isna().any().any():
        raise ValueError("adaptive_weights contains NaN values.")

    if np.isinf(agent_probabilities.to_numpy()).any():
        raise ValueError("agent_probabilities contains infinite values.")

    if np.isinf(adaptive_weights.to_numpy()).any():
        raise ValueError("adaptive_weights contains infinite values.")


def compute_dominant_pathway_table(
    adaptive_weights: pd.DataFrame,
    config: InterpretabilityConfig = InterpretabilityConfig(),
) -> pd.DataFrame:
    """
    Compute case-level dominant SACU pathway from adaptive weights.
    """

    if adaptive_weights.empty:
        raise ValueError("adaptive_weights is empty.")

    dominant_weight_col = adaptive_weights.idxmax(axis=1)

    dominant_pathway = dominant_weight_col.map(
        lambda x: _agent_name_from_weight_column(x, config)
    )

    return pd.DataFrame(
        {
            "dominant_pathway": dominant_pathway,
            "dominant_weight": adaptive_weights.max(axis=1),
        },
        index=adaptive_weights.index,
    )


def compute_pathway_weight_distribution(
    adaptive_weights: pd.DataFrame,
    config: InterpretabilityConfig = InterpretabilityConfig(),
) -> pd.DataFrame:
    """
    Summarize adaptive influence-weight distribution per pathway.
    """

    rows: List[Dict[str, float | str | int]] = []

    dominant_table = compute_dominant_pathway_table(
        adaptive_weights,
        config,
    )

    for column in adaptive_weights.columns:
        pathway = _agent_name_from_weight_column(
            column,
            config,
        )

        values = adaptive_weights[column].astype(float)

        rows.append(
            {
                "pathway": pathway,
                "mean_adaptive_weight": float(values.mean()),
                "std_adaptive_weight": float(values.std()),
                "median_adaptive_weight": float(values.median()),
                "min_adaptive_weight": float(values.min()),
                "max_adaptive_weight": float(values.max()),
                "dominant_cases": int(
                    (dominant_table["dominant_pathway"] == pathway).sum()
                ),
                "dominant_cases_fraction": float(
                    (dominant_table["dominant_pathway"] == pathway).mean()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("mean_adaptive_weight", ascending=False)
        .reset_index(drop=True)
    )


def compute_individual_pathway_performance(
    y_true: np.ndarray | pd.Series,
    agent_probabilities: pd.DataFrame,
    config: InterpretabilityConfig = InterpretabilityConfig(),
) -> pd.DataFrame:
    """
    Compute standalone discriminative performance for each SACU pathway.
    """

    rows: List[Dict[str, float | str]] = []

    for column in agent_probabilities.columns:
        pathway = _agent_name_from_probability_column(
            column,
            config,
        )

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_probability=agent_probabilities[column],
            threshold=config.threshold,
            model_name=pathway,
        )

        metrics["pathway"] = pathway
        rows.append(metrics)

    return pd.DataFrame(rows)


def build_pathway_contribution_summary(
    weight_distribution: pd.DataFrame,
    individual_performance: Optional[pd.DataFrame] = None,
    ablation_loss: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Combine interpretability evidence into one pathway-contribution table.
    """

    if weight_distribution.empty:
        return pd.DataFrame()

    summary = weight_distribution.copy()

    if individual_performance is not None and not individual_performance.empty:
        perf_cols = [
            "pathway",
            "roc_auc",
            "average_precision",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1",
        ]

        available = [
            c for c in perf_cols
            if c in individual_performance.columns
        ]

        summary = summary.merge(
            individual_performance[available],
            on="pathway",
            how="left",
        )

    if ablation_loss is not None and not ablation_loss.empty:
        loss = ablation_loss.copy()

        if "removed_agent" in loss.columns:
            loss = loss.rename(columns={"removed_agent": "pathway"})

        loss_cols = [
            "pathway",
            "delta_roc_auc",
            "delta_average_precision",
            "delta_balanced_accuracy",
            "delta_f1",
        ]

        available = [
            c for c in loss_cols
            if c in loss.columns
        ]

        summary = summary.merge(
            loss[available],
            on="pathway",
            how="left",
        )

    return summary.reset_index(drop=True)


def analyze_pathway_contributions(
    y_true: np.ndarray | pd.Series,
    agent_probabilities: pd.DataFrame,
    adaptive_weights: pd.DataFrame,
    ablation_loss: Optional[pd.DataFrame] = None,
    config: InterpretabilityConfig = InterpretabilityConfig(),
) -> Dict[str, pd.DataFrame]:
    """
    Run complete organizational interpretability analysis.
    """

    _validate_frames(
        agent_probabilities,
        adaptive_weights,
    )

    weight_distribution = compute_pathway_weight_distribution(
        adaptive_weights,
        config,
    )

    dominant_table = compute_dominant_pathway_table(
        adaptive_weights,
        config,
    )

    individual_performance = compute_individual_pathway_performance(
        y_true=y_true,
        agent_probabilities=agent_probabilities,
        config=config,
    )

    contribution_summary = build_pathway_contribution_summary(
        weight_distribution=weight_distribution,
        individual_performance=individual_performance,
        ablation_loss=ablation_loss,
    )

    correctness_table = compute_pathway_correctness_association(
        y_true=y_true,
        agent_probabilities=agent_probabilities,
        adaptive_weights=adaptive_weights,
        config=config,
    )

    evidence_table = build_interpretability_evidence_table(
        contribution_summary=contribution_summary,
        correctness_table=correctness_table,
    )

    return {
        "pathway_contribution_summary": contribution_summary,
        "adaptive_weight_distribution": weight_distribution,
        "case_level_dominant_pathway": dominant_table,
        "individual_pathway_performance": individual_performance,
        "pathway_correctness_association": correctness_table,
        "interpretability_evidence_table": evidence_table,
    }


def compute_pathway_correctness_association(
    y_true: np.ndarray | pd.Series,
    agent_probabilities: pd.DataFrame,
    adaptive_weights: pd.DataFrame,
    config: InterpretabilityConfig = InterpretabilityConfig(),
) -> pd.DataFrame:
    """
    Analyze whether dominant pathway decisions align with correctness.
    """

    _validate_frames(
        agent_probabilities,
        adaptive_weights,
    )

    y = np.asarray(y_true).astype(int)

    if len(y) != len(agent_probabilities):
        raise ValueError("y_true and agent_probabilities have inconsistent lengths.")

    dominant = compute_dominant_pathway_table(
        adaptive_weights,
        config,
    )

    rows = []

    for pathway, group_idx in dominant.groupby("dominant_pathway").groups.items():
        idx = list(group_idx)

        probability_col = f"{pathway}{config.probability_suffix}"

        if probability_col not in agent_probabilities.columns:
            continue

        probs = agent_probabilities.loc[idx, probability_col]
        preds = (probs >= config.threshold).astype(int)

        correct = preds.to_numpy() == y[
            agent_probabilities.index.get_indexer(idx)
        ]

        rows.append(
            {
                "pathway": pathway,
                "dominant_cases": int(len(idx)),
                "dominant_case_accuracy": float(np.mean(correct)) if len(idx) else np.nan,
                "mean_dominant_weight": float(
                    dominant.loc[idx, "dominant_weight"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_interpretability_evidence_table(
    contribution_summary: pd.DataFrame,
    correctness_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build compact evidence table for manuscript/reviewer response.
    """

    if contribution_summary.empty:
        return pd.DataFrame()

    table = contribution_summary.copy()

    if not correctness_table.empty:
        table = table.merge(
            correctness_table,
            on="pathway",
            how="left",
            suffixes=("", "_dominant"),
        )

    keep_cols = [
        "pathway",
        "mean_adaptive_weight",
        "dominant_cases",
        "dominant_cases_fraction",
        "roc_auc",
        "balanced_accuracy",
        "delta_roc_auc",
        "delta_balanced_accuracy",
        "dominant_case_accuracy",
    ]

    available = [c for c in keep_cols if c in table.columns]

    return table[available].copy()


def summarize_interpretability(
    interpretability_outputs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Summarize organizational interpretability findings.
    """

    contribution = interpretability_outputs.get(
        "pathway_contribution_summary",
        pd.DataFrame(),
    )

    if contribution.empty:
        return pd.DataFrame(columns=["metric", "value"])

    rows = []

    if "mean_adaptive_weight" in contribution.columns:
        highest_weight = contribution.sort_values(
            "mean_adaptive_weight",
            ascending=False,
        ).iloc[0]

        rows.append(
            {
                "metric": "highest_mean_adaptive_weight_pathway",
                "value": highest_weight["pathway"],
            }
        )

        rows.append(
            {
                "metric": "highest_mean_adaptive_weight",
                "value": float(highest_weight["mean_adaptive_weight"]),
            }
        )

    if "dominant_cases_fraction" in contribution.columns:
        dominant = contribution.sort_values(
            "dominant_cases_fraction",
            ascending=False,
        ).iloc[0]

        rows.append(
            {
                "metric": "most_frequent_dominant_pathway",
                "value": dominant["pathway"],
            }
        )

        rows.append(
            {
                "metric": "dominant_cases_fraction",
                "value": float(dominant["dominant_cases_fraction"]),
            }
        )

    if "roc_auc" in contribution.columns:
        best_standalone = contribution.sort_values(
            "roc_auc",
            ascending=False,
        ).iloc[0]

        rows.append(
            {
                "metric": "best_standalone_pathway",
                "value": best_standalone["pathway"],
            }
        )

        rows.append(
            {
                "metric": "best_standalone_roc_auc",
                "value": float(best_standalone["roc_auc"]),
            }
        )

    if "delta_roc_auc" in contribution.columns:
        largest_loss = contribution.sort_values(
            "delta_roc_auc",
            ascending=False,
        ).iloc[0]

        rows.append(
            {
                "metric": "largest_ablation_loss_pathway",
                "value": largest_loss["pathway"],
            }
        )

        rows.append(
            {
                "metric": "largest_ablation_delta_roc_auc",
                "value": float(largest_loss["delta_roc_auc"]),
            }
        )

    return pd.DataFrame(rows)
