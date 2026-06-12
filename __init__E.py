# src/organizational_sacu_mammography/evaluation/__init__.py
"""
Evaluation utilities for Organizational-SACU-Mammography.

This package implements the complete evaluation framework used by the
Organizational SACU architecture, including:

- Classification metrics
- Clinical threshold analysis
- Organizational ablation studies
- Bootstrap confidence intervals
- Organizational interpretability analysis
- Coordination-mechanism validation
"""

from .metrics import (
    compute_classification_metrics,
    compute_confusion_statistics,
    summarize_metrics,
)

from .threshold_analysis import (
    ThresholdAnalysisConfig,
    analyze_operating_points,
    summarize_threshold_analysis,
)

from .ablation import (
    AblationConfig,
    run_pathway_ablation,
    summarize_ablation_results,
)

from .bootstrap_ci import (
    BootstrapCIConfig,
    compute_bootstrap_confidence_intervals,
    summarize_bootstrap_results,
)

from .interpretability import (
    InterpretabilityConfig,
    analyze_pathway_contributions,
    summarize_interpretability,
)

from .coordination_ablation import (
    CoordinationAblationConfig,
    evaluate_coordination_mechanisms,
    summarize_coordination_ablation,
)

__all__ = [
    "compute_classification_metrics",
    "compute_confusion_statistics",
    "summarize_metrics",
    "ThresholdAnalysisConfig",
    "analyze_operating_points",
    "summarize_threshold_analysis",
    "AblationConfig",
    "run_pathway_ablation",
    "summarize_ablation_results",
    "BootstrapCIConfig",
    "compute_bootstrap_confidence_intervals",
    "summarize_bootstrap_results",
    "InterpretabilityConfig",
    "analyze_pathway_contributions",
    "summarize_interpretability",
    "CoordinationAblationConfig",
    "evaluate_coordination_mechanisms",
    "summarize_coordination_ablation",
]
