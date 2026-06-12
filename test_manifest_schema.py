# tests/test_manifest_schema.py
from __future__ import annotations

import pandas as pd
import pytest

from organizational_sacu_mammography.representation.manifest_builder import (
    ManifestSplitConfig,
    build_modeling_manifests,
    summarize_modeling_manifest,
    verify_patient_level_separation,
)


REQUIRED_MANIFEST_COLUMNS = {
    "patient_id",
    "study_id",
    "image_id",
    "image_path",
    "view",
    "laterality",
    "acquisition_date",
    "diagnosis_label",
}


def _sample_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [
                "P001",
                "P001",
                "P002",
                "P002",
                "P003",
                "P003",
                "P004",
                "P004",
                "P005",
                "P005",
                "P006",
                "P006",
            ],
            "study_id": [
                "S001",
                "S001",
                "S002",
                "S002",
                "S003",
                "S003",
                "S004",
                "S004",
                "S005",
                "S005",
                "S006",
                "S006",
            ],
            "image_id": [
                f"I{i:03d}"
                for i in range(12)
            ],
            "image_path": [
                f"raw/P{i:03d}.dcm"
                for i in range(12)
            ],
            "view": [
                "LCC",
                "LMLO",
                "RCC",
                "RMLO",
                "LCC",
                "LMLO",
                "RCC",
                "RMLO",
                "LCC",
                "LMLO",
                "RCC",
                "RMLO",
            ],
            "laterality": [
                "L",
                "L",
                "R",
                "R",
                "L",
                "L",
                "R",
                "R",
                "L",
                "L",
                "R",
                "R",
            ],
            "acquisition_date": [
                "2020-01-01",
            ]
            * 12,
            "diagnosis_label": [
                0,
                0,
                1,
                1,
                0,
                0,
                1,
                1,
                0,
                0,
                1,
                1,
            ],
        }
    )


def test_manifest_template_contains_required_columns() -> None:
    manifest = _sample_manifest()

    assert REQUIRED_MANIFEST_COLUMNS.issubset(
        set(manifest.columns)
    )


def test_manifest_has_no_missing_required_values() -> None:
    manifest = _sample_manifest()

    assert manifest[
        list(REQUIRED_MANIFEST_COLUMNS)
    ].isna().sum().sum() == 0


def test_manifest_label_is_binary() -> None:
    manifest = _sample_manifest()

    assert set(
        manifest["diagnosis_label"].unique()
    ).issubset({0, 1})


def test_manifest_views_are_valid() -> None:
    manifest = _sample_manifest()

    valid_views = {
        "LCC",
        "LMLO",
        "RCC",
        "RMLO",
        "CC",
        "MLO",
    }

    assert set(
        manifest["view"].unique()
    ).issubset(valid_views)


def test_manifest_laterality_is_valid() -> None:
    manifest = _sample_manifest()

    assert set(
        manifest["laterality"].unique()
    ).issubset({"L", "R"})


def test_build_modeling_manifest_adds_split_column() -> None:
    manifest = _sample_manifest()

    config = ManifestSplitConfig(
        train_fraction=0.50,
        validation_fraction=0.25,
        test_fraction=0.25,
        random_seed=42,
        stratify_by_label=False,
    )

    split_manifest = build_modeling_manifests(
        manifest,
        config=config,
    )

    assert "split" in split_manifest.columns

    assert set(
        split_manifest["split"].unique()
    ).issubset(
        {
            "train",
            "validation",
            "test",
        }
    )


def test_patient_level_split_has_no_leakage() -> None:
    manifest = _sample_manifest()

    config = ManifestSplitConfig(
        train_fraction=0.50,
        validation_fraction=0.25,
        test_fraction=0.25,
        random_seed=42,
        stratify_by_label=False,
    )

    split_manifest = build_modeling_manifests(
        manifest,
        config=config,
    )

    conflicts = verify_patient_level_separation(
        split_manifest,
        config=config,
    )

    assert conflicts.empty


def test_manifest_summary_is_not_empty() -> None:
    manifest = _sample_manifest()

    config = ManifestSplitConfig(
        train_fraction=0.50,
        validation_fraction=0.25,
        test_fraction=0.25,
        random_seed=42,
        stratify_by_label=False,
    )

    split_manifest = build_modeling_manifests(
        manifest,
        config=config,
    )

    summary = summarize_modeling_manifest(
        split_manifest,
        config=config,
    )

    assert not summary.empty
    assert {
        "split",
        "n_records",
        "n_patients",
    }.issubset(summary.columns)


def test_manifest_builder_rejects_missing_required_column() -> None:
    manifest = _sample_manifest().drop(
        columns=["patient_id"]
    )

    with pytest.raises(KeyError):
        build_modeling_manifests(manifest)
