# src/organizational_sacu_mammography/models/sacu_agents.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .shallow_agent import (
    ShallowAgentConfig,
    ShallowDiagnosticAgent,
    summarize_agent_collection,
)


DEFAULT_SACU_AGENT_NAMES = [
    "LocalRegionalAgent",
    "MultiViewAgent",
    "BilateralAgent",
    "TemporalSpatialAgent",
    "MetadataAgent",
    "AdaptiveControlAgent",
]


DEFAULT_PATHWAY_PREFIXES = {
    "LocalRegionalAgent": [
        "local_",
        "regional_",
        "lr_",
        "left_cc_",
        "left_mlo_",
        "right_cc_",
        "right_mlo_",
    ],
    "MultiViewAgent": [
        "multiview_",
        "cc_mlo_",
        "view_pair_",
    ],
    "BilateralAgent": [
        "bilateral_",
        "asymmetry_",
        "left_right_",
    ],
    "TemporalSpatialAgent": [
        "temporal_",
        "spatial_",
        "progression_",
        "prior_",
        "gap_",
    ],
    "MetadataAgent": [
        "metadata_",
        "age",
        "density",
        "manufacturer",
        "laterality",
    ],
    "AdaptiveControlAgent": [
        "mask_",
        "availability_",
        "completeness_",
        "active_pathway_",
        "complexity_",
    ],
}


@dataclass
class SACUAgentConfig:
    """
    Configuration for constructing organizational SACU agents.
    """

    agent_type: str = "gradient_boosting"
    random_seed: int = 42
    probability_threshold: float = 0.50

    n_estimators: int = 300
    max_depth: Optional[int] = 3
    learning_rate: float = 0.05

    logistic_c: float = 1.0
    logistic_max_iter: int = 5000

    agent_names: List[str] = field(
        default_factory=lambda: DEFAULT_SACU_AGENT_NAMES.copy()
    )

    pathway_prefixes: Mapping[str, List[str]] = field(
        default_factory=lambda: DEFAULT_PATHWAY_PREFIXES.copy()
    )

    explicit_feature_groups: Optional[Mapping[str, List[str]]] = None
    allow_empty_agent_features: bool = False


def _validate_sacu_config(config: SACUAgentConfig) -> None:
    if not config.agent_names:
        raise ValueError("At least one SACU agent must be configured.")

    duplicated = [
        name for name in config.agent_names if config.agent_names.count(name) > 1
    ]

    if duplicated:
        raise ValueError(f"Duplicate SACU agent names found: {sorted(set(duplicated))}")


def infer_agent_feature_groups(
    feature_names: List[str],
    config: SACUAgentConfig = SACUAgentConfig(),
) -> Dict[str, List[str]]:
    """
    Infer pathway-specific feature groups from feature-name prefixes.
    """

    _validate_sacu_config(config)

    if config.explicit_feature_groups is not None:
        groups = {
            agent: [
                feature for feature in features if feature in feature_names
            ]
            for agent, features in config.explicit_feature_groups.items()
        }

        for agent in config.agent_names:
            groups.setdefault(agent, [])

        return groups

    groups: Dict[str, List[str]] = {}

    lowered_features = {
        feature: feature.lower()
        for feature in feature_names
    }

    for agent_name in config.agent_names:
        prefixes = [
            prefix.lower()
            for prefix in config.pathway_prefixes.get(agent_name, [])
        ]

        matched = []

        for feature, lowered in lowered_features.items():
            if any(lowered.startswith(prefix) or prefix in lowered for prefix in prefixes):
                matched.append(feature)

        groups[agent_name] = matched

    return groups


def build_sacu_agents(
    feature_names: List[str],
    config: SACUAgentConfig = SACUAgentConfig(),
) -> Dict[str, ShallowDiagnosticAgent]:
    """
    Build unfitted role-conditioned SACU agents.
    """

    _validate_sacu_config(config)

    feature_groups = infer_agent_feature_groups(
        feature_names=feature_names,
        config=config,
    )

    agents: Dict[str, ShallowDiagnosticAgent] = {}

    for agent_name in config.agent_names:
        group_features = feature_groups.get(agent_name, [])

        if not group_features and not config.allow_empty_agent_features:
            raise ValueError(
                f"No features were assigned to {agent_name}. "
                "Check feature prefixes or provide explicit_feature_groups."
            )

        agent_config = ShallowAgentConfig(
            agent_name=agent_name,
            agent_type=config.agent_type,
            random_seed=config.random_seed,
            probability_threshold=config.probability_threshold,
            logistic_c=config.logistic_c,
            logistic_max_iter=config.logistic_max_iter,
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            feature_names=group_features,
        )

        agents[agent_name] = ShallowDiagnosticAgent(
            config=agent_config,
        )

    return agents


def _subset_matrix(
    X: pd.DataFrame,
    features: List[str],
) -> pd.DataFrame:
    missing = [feature for feature in features if feature not in X.columns]

    if missing:
        raise KeyError(f"Missing agent features: {missing}")

    return X.loc[:, features].copy()


def train_sacu_agents(
    agents: Dict[str, ShallowDiagnosticAgent],
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
) -> Dict[str, ShallowDiagnosticAgent]:
    """
    Train all SACU agents on their pathway-specific feature subsets.
    """

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")

    trained_agents: Dict[str, ShallowDiagnosticAgent] = {}

    for agent_name, agent in agents.items():
        features = agent.config.feature_names

        if not features:
            continue

        X_agent = _subset_matrix(
            X_train,
            features,
        )

        trained_agents[agent_name] = agent.fit(
            X_agent,
            y_train,
        )

    return trained_agents


def evaluate_sacu_agents(
    agents: Dict[str, ShallowDiagnosticAgent],
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate all fitted SACU agents.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    rows = []

    for agent_name, agent in agents.items():
        if not agent.is_fitted:
            continue

        X_agent = _subset_matrix(
            X,
            agent.config.feature_names,
        )

        metrics = agent.evaluate(
            X_agent,
            y,
        )

        metrics["n_features"] = len(agent.config.feature_names)
        rows.append(metrics)

    return pd.DataFrame(rows)


def predict_sacu_agent_probabilities(
    agents: Dict[str, ShallowDiagnosticAgent],
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate one probability column per SACU agent.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    output: Dict[str, np.ndarray] = {}

    for agent_name, agent in agents.items():
        if not agent.is_fitted:
            continue

        X_agent = _subset_matrix(
            X,
            agent.config.feature_names,
        )

        output[f"{agent_name}_probability"] = agent.predict_proba(
            X_agent,
        )

    return pd.DataFrame(output, index=X.index)


def predict_sacu_agent_labels(
    agents: Dict[str, ShallowDiagnosticAgent],
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate one binary prediction column per SACU agent.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    output: Dict[str, np.ndarray] = {}

    for agent_name, agent in agents.items():
        if not agent.is_fitted:
            continue

        X_agent = _subset_matrix(
            X,
            agent.config.feature_names,
        )

        output[f"{agent_name}_prediction"] = agent.predict(
            X_agent,
        )

    return pd.DataFrame(output, index=X.index)


def collect_sacu_feature_importance(
    agents: Dict[str, ShallowDiagnosticAgent],
) -> pd.DataFrame:
    """
    Collect pathway-level feature-importance evidence from fitted agents.
    """

    frames = []

    for agent in agents.values():
        if agent.feature_importance_ is not None:
            frames.append(agent.feature_importance_)

    if not frames:
        return pd.DataFrame(
            columns=[
                "agent_name",
                "feature",
                "importance",
            ]
        )

    return pd.concat(
        frames,
        axis=0,
        ignore_index=True,
    )


def summarize_sacu_agents(
    agents: Dict[str, ShallowDiagnosticAgent],
) -> pd.DataFrame:
    """
    Summarize SACU role-conditioned agents.
    """

    summary = summarize_agent_collection(agents)

    if summary.empty:
        return summary

    summary["n_features"] = summary["agent_name"].map(
        {
            name: len(agent.config.feature_names)
            for name, agent in agents.items()
        }
    )

    return summary


def summarize_sacu_feature_groups(
    agents: Dict[str, ShallowDiagnosticAgent],
) -> pd.DataFrame:
    """
    Summarize feature allocation across organizational roles.
    """

    rows = []

    for agent_name, agent in agents.items():
        rows.append(
            {
                "agent_name": agent_name,
                "n_features": len(agent.config.feature_names),
                "agent_type": agent.config.agent_type,
                "is_fitted": agent.is_fitted,
            }
        )

    return pd.DataFrame(rows)
