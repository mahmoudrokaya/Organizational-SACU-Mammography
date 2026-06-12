# src/organizational_sacu_mammography/models/training.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .sacu_agents import (
    SACUAgentConfig,
    build_sacu_agents,
    train_sacu_agents,
    evaluate_sacu_agents,
    predict_sacu_agent_probabilities,
    collect_sacu_feature_importance,
    summarize_sacu_agents,
    summarize_sacu_feature_groups,
)

from .coordination import (
    CoordinationConfig,
    compute_agent_reliability,
    compute_adaptive_agent_weights,
    apply_coordination_weights,
    summarize_coordination_weights,
    audit_coordination_outputs,
)

from .fusion import (
    FusionConfig,
    build_fusion_feature_matrix,
    train_fusion_model,
    predict_fusion_probabilities,
    evaluate_fusion_model,
    extract_fusion_coefficients,
    summarize_fusion_model,
    export_fusion_predictions,
)


@dataclass
class TrainingConfig:
    """
    End-to-end configuration for the Organizational SACU modeling pipeline.
    """

    sacu_config: SACUAgentConfig = SACUAgentConfig()
    coordination_config: CoordinationConfig = CoordinationConfig()
    fusion_config: FusionConfig = FusionConfig()

    label_col: str = "diagnosis_label"
    record_id_col: Optional[str] = None

    use_validation_reliability: bool = True
    export_case_level_audit: bool = True


def _validate_matrix(
    X: pd.DataFrame,
    name: str,
) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")

    if X.empty:
        raise ValueError(f"{name} is empty.")

    if X.isna().any().any():
        raise ValueError(f"{name} contains NaN values.")

    if np.isinf(X.to_numpy()).any():
        raise ValueError(f"{name} contains infinite values.")


def _validate_labels(
    y: pd.Series | np.ndarray,
    expected_length: int,
    name: str,
) -> None:
    if len(y) != expected_length:
        raise ValueError(
            f"{name} length does not match feature matrix length."
        )

    values = pd.Series(y).dropna().unique()

    if len(values) < 2:
        raise ValueError(
            f"{name} must contain at least two classes."
        )


def train_organizational_sacu_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: pd.Series | np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    pathway_masks_train: Optional[pd.DataFrame] = None,
    pathway_masks_validation: Optional[pd.DataFrame] = None,
    pathway_masks_test: Optional[pd.DataFrame] = None,
    config: TrainingConfig = TrainingConfig(),
) -> Dict[str, object]:
    """
    Train and evaluate the complete Organizational SACU pipeline.

    Pipeline stages:
    1. Build role-conditioned shallow SACU agents.
    2. Train each agent on pathway-specific feature subsets.
    3. Estimate validation reliability.
    4. Compute adaptive coordination weights.
    5. Train learned shallow meta-fusion.
    6. Evaluate final SACU fusion on the independent test set.
    """

    _validate_matrix(X_train, "X_train")
    _validate_matrix(X_validation, "X_validation")
    _validate_matrix(X_test, "X_test")

    _validate_labels(y_train, len(X_train), "y_train")
    _validate_labels(y_validation, len(X_validation), "y_validation")
    _validate_labels(y_test, len(X_test), "y_test")

    feature_names = X_train.columns.tolist()

    agents = build_sacu_agents(
        feature_names=feature_names,
        config=config.sacu_config,
    )

    agents = train_sacu_agents(
        agents=agents,
        X_train=X_train,
        y_train=y_train,
    )

    train_agent_probabilities = predict_sacu_agent_probabilities(
        agents,
        X_train,
    )

    validation_agent_probabilities = predict_sacu_agent_probabilities(
        agents,
        X_validation,
    )

    test_agent_probabilities = predict_sacu_agent_probabilities(
        agents,
        X_test,
    )

    if config.use_validation_reliability:
        reliability_df = compute_agent_reliability(
            validation_probabilities=validation_agent_probabilities,
            y_validation=y_validation,
            config=config.coordination_config,
        )
    else:
        reliability_df = None

    train_weights = compute_adaptive_agent_weights(
        agent_probabilities=train_agent_probabilities,
        reliability_df=reliability_df,
        pathway_masks=pathway_masks_train,
        config=config.coordination_config,
    )

    validation_weights = compute_adaptive_agent_weights(
        agent_probabilities=validation_agent_probabilities,
        reliability_df=reliability_df,
        pathway_masks=pathway_masks_validation,
        config=config.coordination_config,
    )

    test_weights = compute_adaptive_agent_weights(
        agent_probabilities=test_agent_probabilities,
        reliability_df=reliability_df,
        pathway_masks=pathway_masks_test,
        config=config.coordination_config,
    )

    train_coordinated_probability = apply_coordination_weights(
        agent_probabilities=train_agent_probabilities,
        adaptive_weights=train_weights,
        config=config.coordination_config,
    )

    validation_coordinated_probability = apply_coordination_weights(
        agent_probabilities=validation_agent_probabilities,
        adaptive_weights=validation_weights,
        config=config.coordination_config,
    )

    test_coordinated_probability = apply_coordination_weights(
        agent_probabilities=test_agent_probabilities,
        adaptive_weights=test_weights,
        config=config.coordination_config,
    )

    X_fusion_train = build_fusion_feature_matrix(
        agent_probabilities=train_agent_probabilities,
        coordinated_probability=train_coordinated_probability,
        adaptive_weights=train_weights,
        config=config.fusion_config,
    )

    X_fusion_validation = build_fusion_feature_matrix(
        agent_probabilities=validation_agent_probabilities,
        coordinated_probability=validation_coordinated_probability,
        adaptive_weights=validation_weights,
        config=config.fusion_config,
    )

    X_fusion_test = build_fusion_feature_matrix(
        agent_probabilities=test_agent_probabilities,
        coordinated_probability=test_coordinated_probability,
        adaptive_weights=test_weights,
        config=config.fusion_config,
    )

    fusion_model = train_fusion_model(
        X_fusion_train=X_fusion_train,
        y_train=y_train,
        config=config.fusion_config,
    )

    validation_fusion_probability = predict_fusion_probabilities(
        fusion_model=fusion_model,
        X_fusion=X_fusion_validation,
    )

    test_fusion_probability = predict_fusion_probabilities(
        fusion_model=fusion_model,
        X_fusion=X_fusion_test,
    )

    validation_fusion_metrics = evaluate_fusion_model(
        fusion_model=fusion_model,
        X_fusion=X_fusion_validation,
        y=y_validation,
        config=config.fusion_config,
    )

    test_fusion_metrics = evaluate_fusion_model(
        fusion_model=fusion_model,
        X_fusion=X_fusion_test,
        y=y_test,
        config=config.fusion_config,
    )

    agent_validation_metrics = evaluate_sacu_agents(
        agents=agents,
        X=X_validation,
        y=y_validation,
    )

    agent_test_metrics = evaluate_sacu_agents(
        agents=agents,
        X=X_test,
        y=y_test,
    )

    fusion_coefficients = extract_fusion_coefficients(
        fusion_model=fusion_model,
        feature_names=X_fusion_train.columns.tolist(),
    )

    feature_importance = collect_sacu_feature_importance(
        agents,
    )

    test_predictions = export_fusion_predictions(
        probabilities=test_fusion_probability,
        y_true=y_test,
        threshold=config.fusion_config.probability_threshold,
    )

    if config.record_id_col is not None and config.record_id_col in X_test.columns:
        test_predictions.insert(
            0,
            config.record_id_col,
            X_test[config.record_id_col].values,
        )

    outputs: Dict[str, object] = {
        "agents": agents,
        "fusion_model": fusion_model,
        "reliability": reliability_df,
        "train_agent_probabilities": train_agent_probabilities,
        "validation_agent_probabilities": validation_agent_probabilities,
        "test_agent_probabilities": test_agent_probabilities,
        "train_weights": train_weights,
        "validation_weights": validation_weights,
        "test_weights": test_weights,
        "train_coordinated_probability": train_coordinated_probability,
        "validation_coordinated_probability": validation_coordinated_probability,
        "test_coordinated_probability": test_coordinated_probability,
        "X_fusion_train": X_fusion_train,
        "X_fusion_validation": X_fusion_validation,
        "X_fusion_test": X_fusion_test,
        "validation_fusion_probability": validation_fusion_probability,
        "test_fusion_probability": test_fusion_probability,
        "validation_fusion_metrics": validation_fusion_metrics,
        "test_fusion_metrics": test_fusion_metrics,
        "agent_validation_metrics": agent_validation_metrics,
        "agent_test_metrics": agent_test_metrics,
        "fusion_coefficients": fusion_coefficients,
        "agent_feature_importance": feature_importance,
        "test_predictions": test_predictions,
        "agent_summary": summarize_sacu_agents(agents),
        "agent_feature_groups": summarize_sacu_feature_groups(agents),
        "coordination_summary": summarize_coordination_weights(test_weights),
        "fusion_summary": summarize_fusion_model(
            evaluation_metrics=test_fusion_metrics,
            fusion_coefficients=fusion_coefficients,
        ),
    }

    if config.export_case_level_audit:
        outputs["test_coordination_audit"] = audit_coordination_outputs(
            agent_probabilities=test_agent_probabilities,
            adaptive_weights=test_weights,
            coordinated_probability=test_coordinated_probability,
        )

    return outputs


def summarize_training_outputs(
    outputs: Dict[str, object],
) -> pd.DataFrame:
    """
    Summarize end-to-end Organizational SACU training outputs.
    """

    rows = []

    if "test_fusion_metrics" in outputs:
        for metric, value in outputs["test_fusion_metrics"].items():
            rows.append(
                {
                    "section": "test_fusion_performance",
                    "metric": metric,
                    "value": value,
                }
            )

    if "validation_fusion_metrics" in outputs:
        for metric, value in outputs["validation_fusion_metrics"].items():
            rows.append(
                {
                    "section": "validation_fusion_performance",
                    "metric": metric,
                    "value": value,
                }
            )

    if "agent_test_metrics" in outputs:
        agent_metrics = outputs["agent_test_metrics"]

        if isinstance(agent_metrics, pd.DataFrame) and not agent_metrics.empty:
            best_agent = agent_metrics.sort_values(
                "roc_auc",
                ascending=False,
            ).iloc[0]

            rows.append(
                {
                    "section": "agent_performance",
                    "metric": "best_test_agent",
                    "value": best_agent["agent_name"],
                }
            )

            rows.append(
                {
                    "section": "agent_performance",
                    "metric": "best_test_agent_roc_auc",
                    "value": float(best_agent["roc_auc"]),
                }
            )

    if "coordination_summary" in outputs:
        coordination_summary = outputs["coordination_summary"]

        if isinstance(coordination_summary, pd.DataFrame) and not coordination_summary.empty:
            dominant = coordination_summary.iloc[0]

            rows.append(
                {
                    "section": "coordination",
                    "metric": "highest_mean_weight_agent",
                    "value": dominant["agent"],
                }
            )

            rows.append(
                {
                    "section": "coordination",
                    "metric": "highest_mean_weight",
                    "value": float(dominant["mean_weight"]),
                }
            )

    return pd.DataFrame(rows)


def save_training_outputs(
    outputs: Dict[str, object],
    output_dir: str,
) -> Dict[str, str]:
    """
    Save key SACU training outputs as CSV files.
    """

    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    export_map = {
        "agent_validation_metrics": "sacu_agent_validation_metrics.csv",
        "agent_test_metrics": "sacu_agent_test_metrics.csv",
        "fusion_coefficients": "sacu_fusion_coefficients.csv",
        "agent_feature_importance": "sacu_agent_feature_importance.csv",
        "test_predictions": "sacu_test_predictions.csv",
        "agent_summary": "sacu_agent_summary.csv",
        "agent_feature_groups": "sacu_agent_feature_groups.csv",
        "coordination_summary": "sacu_coordination_summary.csv",
        "fusion_summary": "sacu_fusion_summary.csv",
        "test_coordination_audit": "sacu_test_coordination_audit.csv",
    }

    written: Dict[str, str] = {}

    for key, filename in export_map.items():
        value = outputs.get(key)

        if isinstance(value, pd.DataFrame):
            file_path = output_path / filename
            value.to_csv(file_path, index=False)
            written[key] = str(file_path)

    summary = summarize_training_outputs(outputs)

    if not summary.empty:
        file_path = output_path / "sacu_training_summary.csv"
        summary.to_csv(file_path, index=False)
        written["training_summary"] = str(file_path)

    return written
