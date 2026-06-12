# tests/test_imports.py
from __future__ import annotations


def test_top_level_package_imports() -> None:
    import organizational_sacu_mammography

    assert organizational_sacu_mammography is not None


def test_preprocessing_imports() -> None:
    from organizational_sacu_mammography import preprocessing

    assert preprocessing is not None


def test_metadata_imports() -> None:
    from organizational_sacu_mammography import metadata

    assert metadata is not None


def test_representation_imports() -> None:
    from organizational_sacu_mammography import representation

    assert representation is not None


def test_features_imports() -> None:
    from organizational_sacu_mammography import features

    assert features is not None


def test_models_imports() -> None:
    from organizational_sacu_mammography import models

    assert models is not None


def test_evaluation_imports() -> None:
    from organizational_sacu_mammography import evaluation

    assert evaluation is not None


def test_utils_imports() -> None:
    from organizational_sacu_mammography import utils

    assert utils is not None


def test_core_model_classes_import() -> None:
    from organizational_sacu_mammography.models import (
        BaselineModelConfig,
        ShallowAgentConfig,
        SACUAgentConfig,
        CoordinationConfig,
        FusionConfig,
        TrainingConfig,
    )

    assert BaselineModelConfig is not None
    assert ShallowAgentConfig is not None
    assert SACUAgentConfig is not None
    assert CoordinationConfig is not None
    assert FusionConfig is not None
    assert TrainingConfig is not None


def test_core_evaluation_functions_import() -> None:
    from organizational_sacu_mammography.evaluation import (
        compute_classification_metrics,
        analyze_operating_points,
        run_pathway_ablation,
        compute_bootstrap_confidence_intervals,
        analyze_pathway_contributions,
        evaluate_coordination_mechanisms,
    )

    assert compute_classification_metrics is not None
    assert analyze_operating_points is not None
    assert run_pathway_ablation is not None
    assert compute_bootstrap_confidence_intervals is not None
    assert analyze_pathway_contributions is not None
    assert evaluate_coordination_mechanisms is not None
