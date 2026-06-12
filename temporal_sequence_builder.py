# src/organizational_sacu_mammography/representation/temporal_sequence_builder.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class TemporalSequenceConfig:
    """
    Configuration for longitudinal mammography sequence construction.

    Input is expected to be an examination-level table where each row
    represents one patient examination.
    """

    patient_col: str = "patient_id"
    study_col: str = "study_id"
    exam_date_col: str = "exam_date"
    label_col: str = "diagnosis_label"

    allow_single_exam_patients: bool = True
    minimum_exams_for_temporal_case: int = 2
    date_format: str = "%Y-%m-%d"


def _validate_columns(
    df: pd.DataFrame,
    config: TemporalSequenceConfig,
) -> None:
    required = [
        config.patient_col,
        config.study_col,
        config.exam_date_col,
        config.label_col,
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(
            f"Missing required columns for temporal sequence construction: {missing}"
        )


def build_temporal_sequences(
    exam_df: pd.DataFrame,
    config: TemporalSequenceConfig = TemporalSequenceConfig(),
) -> pd.DataFrame:
    """
    Build chronological patient-level temporal examination sequences.

    Returns one row per examination with references to prior and next studies.
    """

    _validate_columns(exam_df, config)

    df = exam_df.copy()

    df[config.exam_date_col] = pd.to_datetime(
        df[config.exam_date_col],
        errors="coerce",
    )

    if df[config.exam_date_col].isna().any():
        n_bad = int(df[config.exam_date_col].isna().sum())
        raise ValueError(
            f"Found {n_bad} records with invalid examination dates."
        )

    df = df.sort_values(
        [config.patient_col, config.exam_date_col, config.study_col]
    ).reset_index(drop=True)

    records: List[Dict] = []

    for patient_id, group in df.groupby(config.patient_col):
        group = group.sort_values(
            [config.exam_date_col, config.study_col]
        ).reset_index(drop=True)

        total_exams = len(group)

        if (
            not config.allow_single_exam_patients
            and total_exams < config.minimum_exams_for_temporal_case
        ):
            continue

        for i, row in group.iterrows():
            has_prior = i > 0
            has_next = i < total_exams - 1

            prior_row = group.iloc[i - 1] if has_prior else None
            next_row = group.iloc[i + 1] if has_next else None

            exam_date = row[config.exam_date_col]

            prior_exam_date = (
                prior_row[config.exam_date_col]
                if prior_row is not None
                else pd.NaT
            )

            next_exam_date = (
                next_row[config.exam_date_col]
                if next_row is not None
                else pd.NaT
            )

            temporal_gap_days = (
                int((exam_date - prior_exam_date).days)
                if has_prior
                else None
            )

            time_to_next_exam_days = (
                int((next_exam_date - exam_date).days)
                if has_next
                else None
            )

            temporal_case = int(
                total_exams >= config.minimum_exams_for_temporal_case
                and has_prior
            )

            records.append(
                {
                    "patient_id": patient_id,
                    "study_id": row[config.study_col],
                    "exam_date": exam_date.strftime(config.date_format),
                    "diagnosis_label": row[config.label_col],

                    "temporal_exam_index": int(i),
                    "total_exams_for_patient": int(total_exams),

                    "has_prior_exam": int(has_prior),
                    "has_next_exam": int(has_next),
                    "temporal_case": int(temporal_case),

                    "prior_study_id": (
                        prior_row[config.study_col]
                        if prior_row is not None
                        else None
                    ),
                    "prior_exam_date": (
                        prior_exam_date.strftime(config.date_format)
                        if has_prior
                        else None
                    ),
                    "temporal_gap_days": temporal_gap_days,

                    "next_study_id": (
                        next_row[config.study_col]
                        if next_row is not None
                        else None
                    ),
                    "next_exam_date": (
                        next_exam_date.strftime(config.date_format)
                        if has_next
                        else None
                    ),
                    "time_to_next_exam_days": time_to_next_exam_days,
                }
            )

    return pd.DataFrame(records)


def summarize_temporal_sequences(
    temporal_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize temporal cohort availability.
    """

    if temporal_df.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "value",
            ]
        )

    patient_counts = (
        temporal_df.groupby("patient_id")["study_id"]
        .count()
        .rename("n_exams")
    )

    temporal_patients = int(
        (patient_counts >= 2).sum()
    )

    non_temporal_patients = int(
        (patient_counts == 1).sum()
    )

    summary = [
        {
            "metric": "total_exams",
            "value": int(len(temporal_df)),
        },
        {
            "metric": "total_patients",
            "value": int(temporal_df["patient_id"].nunique()),
        },
        {
            "metric": "temporal_patients",
            "value": temporal_patients,
        },
        {
            "metric": "non_temporal_patients",
            "value": non_temporal_patients,
        },
        {
            "metric": "temporal_cases",
            "value": int(temporal_df["temporal_case"].sum()),
        },
        {
            "metric": "mean_exams_per_patient",
            "value": float(patient_counts.mean()),
        },
        {
            "metric": "median_exams_per_patient",
            "value": float(patient_counts.median()),
        },
        {
            "metric": "max_exams_per_patient",
            "value": int(patient_counts.max()),
        },
    ]

    if temporal_df["temporal_gap_days"].notna().any():
        summary.extend(
            [
                {
                    "metric": "mean_temporal_gap_days",
                    "value": float(
                        temporal_df["temporal_gap_days"].dropna().mean()
                    ),
                },
                {
                    "metric": "median_temporal_gap_days",
                    "value": float(
                        temporal_df["temporal_gap_days"].dropna().median()
                    ),
                },
            ]
        )

    return pd.DataFrame(summary)


def audit_temporal_sequences(
    temporal_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create an audit table for temporal examination ordering.
    """

    if temporal_df.empty:
        return pd.DataFrame()

    audit = temporal_df[
        [
            "patient_id",
            "study_id",
            "exam_date",
            "temporal_exam_index",
            "total_exams_for_patient",
            "has_prior_exam",
            "prior_study_id",
            "prior_exam_date",
            "temporal_gap_days",
            "temporal_case",
        ]
    ].copy()

    audit["temporal_status"] = "single_exam"

    audit.loc[
        audit["has_prior_exam"] == 1,
        "temporal_status",
    ] = "has_prior_exam"

    audit.loc[
        (audit["total_exams_for_patient"] > 1)
        & (audit["has_prior_exam"] == 0),
        "temporal_status",
    ] = "first_exam_in_sequence"

    return audit
