# src/organizational_sacu_mammography/models/__init__.py
"""
Modeling utilities for Organizational-SACU-Mammography.

This package implements the shallow organizational modeling layer used by the
SACU framework, including:

- Conventional baseline models
- Role-conditioned shallow diagnostic agents
- Specialized SACU agents
- Adaptive coordination mechanisms
- Meta-fusion strategies
- End-to-end training utilities
"""

from .baseline_models import (
    BaselineModelConfig,
    train_baseline_models,
    evaluate_baseline_models,
    summarize_baseline_results,
)

from .shallow_agent import (
    ShallowAgentConfig,
    ShallowDiagnosticAgent,
)

from .sacu_agents import (
    SACUAgentConfig,
    build_sacu_agents,
    train_sacu_agents,
    evaluate_sacu_agents,
    summarize_sacu_agents,
)

from .coordination import (
    CoordinationConfig,
    compute_agent_reliability,
    compute_adaptive_agent_weights,
    summarize_coordination_weights,
)

from .fusion import (
    FusionConfig,
    train_fusion_model,
    predict_fusion_probabilities,
    evaluate_fusion_model,
)

from .training import (
    TrainingConfig,
    train_organizational_sacu_pipeline,
    summarize_training_outputs,
)

__all__ = [
    "BaselineModelConfig",
    "train_baseline_models",
    "evaluate_baseline_models",
    "summarize_baseline_results",
    "ShallowAgentConfig",
    "ShallowDiagnosticAgent",
    "SACUAgentConfig",
    "build_sacu_agents",
    "train_sacu_agents",
    "evaluate_sacu_agents",
    "summarize_sacu_agents",
    "CoordinationConfig",
    "compute_agent_reliability",
    "compute_adaptive_agent_weights",
    "summarize_coordination_weights",
    "FusionConfig",
    "train_fusion_model",
    "predict_fusion_probabilities",
    "evaluate_fusion_model",
    "TrainingConfig",
    "train_organizational_sacu_pipeline",
    "summarize_training_outputs",
]
