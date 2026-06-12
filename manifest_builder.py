# src/organizational_sacu_mammography/representation/manifest_builder.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class ManifestSplitConfig:
    """
    Configuration for patient-level modeling manifest construction.
    """

    patient_col: str = "patient_id"
    study_col: str = "study_id"
    label_col: str = "diagnosis_label"

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    random_seed: int = 42
    stratify_by_label: bool = True

    split_col: str = "split"


def _validate_columns(
    df: pd.DataFrame,
    config: ManifestSplitConfig,
) -> None:
    required = [
        config.patient_col,
        config.study_col,
        config.label_col,
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(
            f"Missing required columns for manifest construction: {missing}"
        )


def _validate_fractions(config: ManifestSplitConfig) -> None:
    total = (
        config.train_fraction
        + config.validation_fraction
        + config.test_fraction
    )

    if not np.isclose(total, 1.0):
        raise ValueError(
            "Train, validation, and test fractions must sum to 1.0. "
            f"Current sum = {total:.4f}"
        )


def _patient_label_table(
    df: pd.DataFrame,
    config: ManifestSplitConfig,
) -> pd.DataFrame:
    """
    Build one label per patient for stratified patient-level splitting.

    If a patient has multiple labels, the maximum label is used so that
    any malignant examination makes the patient positive.
    """

    patient_labels = (
        df.groupby(config.patient_col)[config.label_col]
        .max()
        .reset_index()
        .rename(columns={config.label_col: "patient_label"})
    )

    return patient_labels


def _safe_stratify(labels: pd.Series) -> pd.Series | None:
    """
    Return labels for stratification only when every class has at least two cases.
    """

    counts = labels.value_counts()

    if len(counts) < 2:
        return None

    if (counts < 2).any():
        return None

    return labels


def build_modeling_manifests(
    df: pd.DataFrame,
    config: ManifestSplitConfig = ManifestSplitConfig(),
) -> pd.DataFrame:
    """
    Create train/validation/test modeling manifests using patient-level splitting.

    No patient can appear in more than one partition.
    """

    _validate_columns(df, config)
    _validate_fractions(config)

    patient_labels = _patient_label_table(df, config)

    stratify_first = (
        _safe_stratify(patient_labels["patient_label"])
        if config.stratify_by_label
        else None
    )

    train_patients, temp_patients = train_test_split(
        patient_labels,
        train_size=config.train_fraction,
        random_state=config.random_seed,
        stratify=stratify_first,
    )

    relative_validation_fraction = (
        config.validation_fraction
        / (config.validation_fraction + config.test_fraction)
    )

    stratify_second = (
        _safe_stratify(temp_patients["patient_label"])
        if config.stratify_by_label
        else None
    )

    validation_patients, test_patients = train_test_split(
        temp_patients,
        train_size=relative_validation_fraction,
        random_state=config.random_seed,
        stratify=stratify_second,
    )

    split_map: Dict[str, str] = {}

    for pid in train_patients[config.patient_col]:
        split_map[pid] = "train"

    for pid in validation_patients[config.patient_col]:
        split_map[pid] = "validation"

    for pid in test_patients[config.patient_col]:
        split_map[pid] = "test"

    out = df.copy()
    out[config.split_col] = out[config.patient_col].map(split_map)

    if out[config.split_col].isna().any():
        raise RuntimeError("Some records were not assigned to any split.")

    return out


def summarize_modeling_manifest(
    manifest_df: pd.DataFrame,
    config: ManifestSplitConfig = ManifestSplitConfig(),
) -> pd.DataFrame:
    """
    Summarize patient-level and examination-level split statistics.
    """

    if config.split_col not in manifest_df.columns:
        raise KeyError(f"Missing split column: {config.split_col}")

    records: List[Dict] = []

    for split_name, group in manifest_df.groupby(config.split_col):
        records.append(
            {
                "split": split_name,
                "n_records": int(len(group)),
                "n_patients": int(group[config.patient_col].nunique()),
                "n_studies": int(group[config.study_col].nunique()),
                "positive_records": int((group[config.label_col] == 1).sum()),
                "negative_records": int((group[config.label_col] == 0).sum()),
                "positive_fraction": float(group[config.label_col].mean()),
            }
        )

    return pd.DataFrame(records).sort_values("split").reset_index(drop=True)


def verify_patient_level_separation(
    manifest_df: pd.DataFrame,
    config: ManifestSplitConfig = ManifestSplitConfig(),
) -> pd.DataFrame:
    """
    Verify that no patient appears in more than one split.
    """

    if config.split_col not in manifest_df.columns:
        raise KeyError(f"Missing split column: {config.split_col}")

    patient_split_counts = (
        manifest_df.groupby(config.patient_col)[config.split_col]
        .nunique()
        .reset_index(name="n_splits")
    )

    conflicts = patient_split_counts[
        patient_split_counts["n_splits"] > 1
    ].copy()

    return conflicts


def split_manifest_frames(
    manifest_df: pd.DataFrame,
    config: ManifestSplitConfig = ManifestSplitConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return train, validation, and test dataframes from a manifest.
    """

    if config.split_col not in manifest_df.columns:
        raise KeyError(f"Missing split column: {config.split_col}")

    train_df = manifest_df[
        manifest_df[config.split_col] == "train"
    ].copy()

    validation_df = manifest_df[
        manifest_df[config.split_col] == "validation"
    ].copy()

    test_df = manifest_df[
        manifest_df[config.split_col] == "test"
    ].copy()

    return train_df, validation_df, test_df


def save_manifest_splits(
    manifest_df: pd.DataFrame,
    output_dir: str,
    config: ManifestSplitConfig = ManifestSplitConfig(),
) -> Dict[str, str]:
    """
    Save full manifest and split-specific manifests as CSV files.
    """

    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df, validation_df, test_df = split_manifest_frames(
        manifest_df,
        config=config,
    )

    files = {
        "full_manifest": output_path / "modeling_manifest.csv",
        "train_manifest": output_path / "train_manifest.csv",
        "validation_manifest": output_path / "validation_manifest.csv",
        "test_manifest": output_path / "test_manifest.csv",
    }

    manifest_df.to_csv(files["full_manifest"], index=False)
    train_df.to_csv(files["train_manifest"], index=False)
    validation_df.to_csv(files["validation_manifest"], index=False)
    test_df.to_csv(files["test_manifest"], index=False)

    return {k: str(v) for k, v in files.items()}
