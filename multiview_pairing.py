# src/organizational_sacu_mammography/representation/multiview_pairing.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class MultiViewPairingConfig:
    """
    Configuration for CC-MLO multi-view pairing.
    """

    patient_col: str = "patient_id"
    study_col: str = "study_id"
    exam_date_col: str = "exam_date"

    laterality_col: str = "laterality"
    view_col: str = "view_position"

    image_col: str = "image_id"
    label_col: str = "diagnosis_label"

    require_same_study: bool = True
    allow_partial_exams: bool = True


def _validate_columns(
    df: pd.DataFrame,
    config: MultiViewPairingConfig,
) -> None:
    required = [
        config.patient_col,
        config.study_col,
        config.exam_date_col,
        config.laterality_col,
        config.view_col,
        config.image_col,
        config.label_col,
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(
            f"Missing required columns for multi-view pairing: {missing}"
        )


def _extract_view(
    group: pd.DataFrame,
    laterality: str,
    view: str,
    image_col: str,
):
    subset = group[
        (group["laterality"] == laterality)
        & (group["view_position"] == view)
    ]

    if len(subset) == 0:
        return None

    return subset.iloc[0][image_col]


def build_multiview_pairs(
    metadata_df: pd.DataFrame,
    config: MultiViewPairingConfig = MultiViewPairingConfig(),
) -> pd.DataFrame:
    """
    Construct examination-level multi-view representations.

    Output:
        One row per examination.
    """

    _validate_columns(metadata_df, config)

    df = metadata_df.copy()

    df[config.laterality_col] = (
        df[config.laterality_col]
        .astype(str)
        .str.upper()
    )

    df[config.view_col] = (
        df[config.view_col]
        .astype(str)
        .str.upper()
    )

    grouping_keys = [
        config.patient_col,
        config.study_col,
    ]

    records: List[Dict] = []

    grouped = df.groupby(grouping_keys)

    for (patient_id, study_id), group in grouped:

        left_cc = _extract_view(
            group,
            "LEFT",
            "CC",
            config.image_col,
        )

        left_mlo = _extract_view(
            group,
            "LEFT",
            "MLO",
            config.image_col,
        )

        right_cc = _extract_view(
            group,
            "RIGHT",
            "CC",
            config.image_col,
        )

        right_mlo = _extract_view(
            group,
            "RIGHT",
            "MLO",
            config.image_col,
        )

        available_views = sum(
            [
                left_cc is not None,
                left_mlo is not None,
                right_cc is not None,
                right_mlo is not None,
            ]
        )

        complete_exam = available_views == 4

        if (
            not config.allow_partial_exams
            and not complete_exam
        ):
            continue

        record = {
            "patient_id": patient_id,
            "study_id": study_id,
            "exam_date": group[
                config.exam_date_col
            ].iloc[0],
            "diagnosis_label": group[
                config.label_col
            ].iloc[0],

            "left_cc_image": left_cc,
            "left_mlo_image": left_mlo,
            "right_cc_image": right_cc,
            "right_mlo_image": right_mlo,

            "left_cc_available": int(left_cc is not None),
            "left_mlo_available": int(left_mlo is not None),
            "right_cc_available": int(right_cc is not None),
            "right_mlo_available": int(right_mlo is not None),

            "available_views": available_views,
            "complete_exam": int(complete_exam),
        }

        records.append(record)

    return pd.DataFrame(records)


def summarize_multiview_pairs(
    paired_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate pairing statistics.
    """

    if len(paired_df) == 0:
        return pd.DataFrame()

    total_exams = len(paired_df)

    complete_exams = int(
        paired_df["complete_exam"].sum()
    )

    partial_exams = total_exams - complete_exams

    summary = [
        {
            "metric": "total_exams",
            "value": total_exams,
        },
        {
            "metric": "complete_exams",
            "value": complete_exams,
        },
        {
            "metric": "partial_exams",
            "value": partial_exams,
        },
        {
            "metric": "complete_exam_ratio",
            "value": complete_exams / total_exams,
        },
        {
            "metric": "mean_available_views",
            "value": paired_df[
                "available_views"
            ].mean(),
        },
    ]

    return pd.DataFrame(summary)


def audit_missing_views(
    paired_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Examination-level missing-view audit.
    """

    records = []

    for _, row in paired_df.iterrows():

        missing = []

        if row["left_cc_available"] == 0:
            missing.append("LEFT_CC")

        if row["left_mlo_available"] == 0:
            missing.append("LEFT_MLO")

        if row["right_cc_available"] == 0:
            missing.append("RIGHT_CC")

        if row["right_mlo_available"] == 0:
            missing.append("RIGHT_MLO")

        records.append(
            {
                "patient_id": row["patient_id"],
                "study_id": row["study_id"],
                "complete_exam": row["complete_exam"],
                "missing_views": ";".join(missing),
                "n_missing": len(missing),
            }
        )

    return pd.DataFrame(records)
