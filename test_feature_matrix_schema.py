# tests/test_feature_matrix_schema.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from organizational_sacu_mammography.features.feature_matrix_builder import (
    build_feature_matrix,
    summarize_feature_matrix,
)


def _sample_feature_sources() -> dict[str, pd.DataFrame]:
    index = [
        "S001",
        "S002",
        "S003",
        "S004",
    ]

    local = pd.DataFrame(
        {
            "study_id": index,
            "local_intensity_mean": [
                0.12,
                0.23,
                0.34,
                0.45,
            ],
            "local_gradient_std": [
                0.01,
                0.02,
                0.03,
                0.04,
            ],
        }
    )

    multiview = pd.DataFrame(
        {
            "study_id": index,
            "multiview_cc_mlo_difference": [
                0.05,
                0.04,
                0.07,
                0.08,
            ],
        }
    )

    bilateral = pd.DataFrame(
        {
            "study_id": index,
            "bilateral_asymmetry_score": [
                0.10,
                0.20,
                0.30,
                0.40,
            ],
        }
    )

    temporal = pd.DataFrame(
        {
            "study_id": index,
            "temporal_gap_days": [
                0,
                180,
                360,
                720,
            ],
            "temporal_progression_score": [
                0.00,
                0.10,
                0.20,
                0.30,
            ],
        }
    )

    metadata = pd.DataFrame(
        {
            "study_id": index,
            "metadata_age": [
                45,
                52,
                60,
                67,
            ],
            "diagnosis_label": [
                0,
                1,
                0,
                1,
            ],
        }
    )

    return {
        "local": local,
        "multiview": multiview,
        "bilateral": bilateral,
        "temporal": temporal,
        "metadata": metadata,
    }


def test_feature_matrix_sources_have_study_id() -> None:
    sources = _sample_feature_sources()

    for frame in sources.values():
        assert "study_id" in frame.columns


def test_feature_matrix_numeric_columns_are_valid() -> None:
    sources = _sample_feature_sources()

    for frame in sources.values():
        numeric = frame.select_dtypes(
            include=[np.number]
        )

        assert not numeric.isna().any().any()
        assert not np.isinf(
            numeric.to_numpy()
        ).any()


def test_build_feature_matrix_from_sources() -> None:
    sources = _sample_feature_sources()

    matrix = build_feature_matrix(
        feature_tables=list(sources.values()),
        key_col="study_id",
    )

    assert isinstance(matrix, pd.DataFrame)
    assert len(matrix) == 4
    assert "study_id" in matrix.columns
    assert "diagnosis_label" in matrix.columns


def test_feature_matrix_summary_is_not_empty() -> None:
    sources = _sample_feature_sources()

    matrix = build_feature_matrix(
        feature_tables=list(sources.values()),
        key_col="study_id",
    )

    summary = summarize_feature_matrix(
        matrix,
        label_col="diagnosis_label",
    )

    assert not summary.empty
    assert {
        "metric",
        "value",
    }.issubset(summary.columns)


def test_feature_matrix_has_no_duplicate_columns() -> None:
    sources = _sample_feature_sources()

    matrix = build_feature_matrix(
        feature_tables=list(sources.values()),
        key_col="study_id",
    )

    assert matrix.columns.duplicated().sum() == 0


def test_feature_matrix_rejects_missing_key_column() -> None:
    sources = _sample_feature_sources()

    bad_table = sources["local"].drop(
        columns=["study_id"]
    )

    with pytest.raises(KeyError):
        build_feature_matrix(
            feature_tables=[
                bad_table,
                sources["metadata"],
            ],
            key_col="study_id",
        )


def test_feature_matrix_target_is_binary() -> None:
    sources = _sample_feature_sources()

    matrix = build_feature_matrix(
        feature_tables=list(sources.values()),
        key_col="study_id",
    )

    assert set(
        matrix["diagnosis_label"].unique()
    ).issubset({0, 1})
