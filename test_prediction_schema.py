# tests/test_prediction_schema.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from organizational_sacu_mammography.models.coordination import (
    CoordinationConfig,
    compute_adaptive_agent_weights,
    apply_coordination_weights,
    audit_coordination_outputs,
)

from organizational_sacu_mammography.models.fusion import (
    export_fusion_predictions,
)


def _sample_agent_probabilities() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "LocalRegionalAgent_probability": [
                0.10,
                0.80,
                0.30,
                0.70,
            ],
            "MultiViewAgent_probability": [
                0.20,
                0.75,
                0.35,
                0.65,
            ],
            "BilateralAgent_probability": [
                0.15,
                0.85,
                0.25,
                0.60,
            ],
        }
    )


def _sample_pathway_masks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "local_regional_mask": [
                1,
                1,
                1,
                1,
            ],
            "multiview_mask": [
                1,
                1,
                1,
                1,
            ],
            "bilateral_mask": [
                1,
                1,
                1,
                1,
            ],
        }
    )


def test_agent_probability_schema() -> None:
    probabilities = _sample_agent_probabilities()

    assert not probabilities.empty

    for column in probabilities.columns:
        assert column.endswith("_probability")
        assert probabilities[column].between(
            0,
            1,
        ).all()


def test_compute_adaptive_weights_schema() -> None:
    probabilities = _sample_agent_probabilities()
    masks = _sample_pathway_masks()

    weights = compute_adaptive_agent_weights(
        agent_probabilities=probabilities,
        pathway_masks=masks,
        config=CoordinationConfig(),
    )

    assert not weights.empty

    for column in weights.columns:
        assert column.endswith("_weight")

    assert np.allclose(
        weights.sum(axis=1),
        1.0,
    )


def test_apply_coordination_weights_outputs_probability() -> None:
    probabilities = _sample_agent_probabilities()
    masks = _sample_pathway_masks()

    weights = compute_adaptive_agent_weights(
        agent_probabilities=probabilities,
        pathway_masks=masks,
        config=CoordinationConfig(),
    )

    coordinated = apply_coordination_weights(
        agent_probabilities=probabilities,
        adaptive_weights=weights,
        config=CoordinationConfig(),
    )

    assert isinstance(coordinated, pd.Series)
    assert coordinated.between(
        0,
        1,
    ).all()


def test_coordination_audit_schema() -> None:
    probabilities = _sample_agent_probabilities()
    masks = _sample_pathway_masks()

    weights = compute_adaptive_agent_weights(
        agent_probabilities=probabilities,
        pathway_masks=masks,
        config=CoordinationConfig(),
    )

    coordinated = apply_coordination_weights(
        agent_probabilities=probabilities,
        adaptive_weights=weights,
        config=CoordinationConfig(),
    )

    audit = audit_coordination_outputs(
        agent_probabilities=probabilities,
        adaptive_weights=weights,
        coordinated_probability=coordinated,
    )

    required = {
        "coordinated_probability",
        "coordinated_prediction",
        "dominant_agent",
        "dominant_weight",
        "adaptive_weight_entropy",
    }

    assert required.issubset(
        audit.columns
    )


def test_export_fusion_predictions_schema() -> None:
    probabilities = pd.Series(
        [
            0.10,
            0.80,
            0.30,
            0.70,
        ],
        name="sacu_fusion_probability",
    )

    y_true = np.array(
        [
            0,
            1,
            0,
            1,
        ]
    )

    predictions = export_fusion_predictions(
        probabilities=probabilities,
        y_true=y_true,
        threshold=0.50,
    )

    required = {
        "sacu_fusion_probability",
        "sacu_fusion_prediction",
        "true_label",
    }

    assert required.issubset(
        predictions.columns
    )

    assert predictions[
        "sacu_fusion_probability"
    ].between(
        0,
        1,
    ).all()

    assert set(
        predictions["sacu_fusion_prediction"].unique()
    ).issubset({0, 1})


def test_adaptive_weights_reject_nan_probabilities() -> None:
    probabilities = _sample_agent_probabilities()
    probabilities.loc[0, "LocalRegionalAgent_probability"] = np.nan

    with pytest.raises(ValueError):
        compute_adaptive_agent_weights(
            agent_probabilities=probabilities,
            pathway_masks=_sample_pathway_masks(),
        )
