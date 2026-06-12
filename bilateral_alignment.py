# src/organizational_sacu_mammography/representation/bilateral_alignment.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class BilateralAlignmentConfig:
    """
    Configuration for bilateral mammography alignment.

    The input is expected to be the examination-level table produced by
    multiview_pairing.py.
    """

    patient_col: str = "patient_id"
    study_col: str = "study_id"
    exam_date_col: str = "exam_date"
    label_col: str = "diagnosis_label"

    left_cc_col: str = "left_cc_image"
    left_mlo_col: str = "left_mlo_image"
    right_cc_col: str = "right_cc_image"
    right_mlo_col: str = "right_mlo_image"

    require_cc_pair: bool = False
    require_mlo_pair: bool = False
    allow_partial_bilateral_pairs: bool = True


def _validate_columns(
    df: pd.DataFrame,
    config: BilateralAlignmentConfig,
) -> None:
    required = [
        config.patient_col,
        config.study_col,
        config.exam_date_col,
        config.label_col,
        config.left_cc_col,
        config.left_mlo_col,
        config.right_cc_col,
        config.right_mlo_col,
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(
            f"Missing required columns for bilateral alignment: {missing}"
        )


def _is_available(value) -> bool:
    return pd.notna(value) and str(value).strip() not in {"", "None", "nan", "NaN"}


def build_bilateral_alignment(
    paired_df: pd.DataFrame,
    config: BilateralAlignmentConfig = BilateralAlignmentConfig(),
) -> pd.DataFrame:
    """
    Build examination-level bilateral alignment records.

    Each output row describes whether left-right CC and MLO pairs are available
    for a mammography examination.
    """

    _validate_columns(paired_df, config)

    records: List[Dict] = []

    for _, row in paired_df.iterrows():
        left_cc_available = _is_available(row[config.left_cc_col])
        right_cc_available = _is_available(row[config.right_cc_col])
        left_mlo_available = _is_available(row[config.left_mlo_col])
        right_mlo_available = _is_available(row[config.right_mlo_col])

        cc_bilateral_available = left_cc_available and right_cc_available
        mlo_bilateral_available = left_mlo_available and right_mlo_available

        bilateral_pair_count = int(cc_bilateral_available) + int(mlo_bilateral_available)
        bilateral_available = bilateral_pair_count > 0
        complete_bilateral = cc_bilateral_available and mlo_bilateral_available

        if config.require_cc_pair and not cc_bilateral_available:
            continue

        if config.require_mlo_pair and not mlo_bilateral_available:
            continue

        if not config.allow_partial_bilateral_pairs and not complete_bilateral:
            continue

        records.append(
            {
                "patient_id": row[config.patient_col],
                "study_id": row[config.study_col],
                "exam_date": row[config.exam_date_col],
                "diagnosis_label": row[config.label_col],

                "left_cc_image": row[config.left_cc_col],
                "right_cc_image": row[config.right_cc_col],
                "left_mlo_image": row[config.left_mlo_col],
                "right_mlo_image": row[config.right_mlo_col],

                "cc_bilateral_available": int(cc_bilateral_available),
                "mlo_bilateral_available": int(mlo_bilateral_available),
                "bilateral_available": int(bilateral_available),
                "complete_bilateral": int(complete_bilateral),
                "bilateral_pair_count": bilateral_pair_count,
            }
        )

    return pd.DataFrame(records)


def summarize_bilateral_alignment(
    bilateral_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize bilateral alignment availability.
    """

    if bilateral_df.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "value",
            ]
        )

    total = len(bilateral_df)

    summary = [
        {
            "metric": "total_exams",
            "value": total,
        },
        {
            "metric": "cc_bilateral_available",
            "value": int(bilateral_df["cc_bilateral_available"].sum()),
        },
        {
            "metric": "mlo_bilateral_available",
            "value": int(bilateral_df["mlo_bilateral_available"].sum()),
        },
        {
            "metric": "any_bilateral_available",
            "value": int(bilateral_df["bilateral_available"].sum()),
        },
        {
            "metric": "complete_bilateral",
            "value": int(bilateral_df["complete_bilateral"].sum()),
        },
        {
            "metric": "complete_bilateral_ratio",
            "value": float(bilateral_df["complete_bilateral"].mean()),
        },
        {
            "metric": "mean_bilateral_pair_count",
            "value": float(bilateral_df["bilateral_pair_count"].mean()),
        },
    ]

    return pd.DataFrame(summary)


def audit_bilateral_alignment(
    bilateral_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce an examination-level bilateral availability audit.
    """

    if bilateral_df.empty:
        return pd.DataFrame()

    audit = bilateral_df[
        [
            "patient_id",
            "study_id",
            "exam_date",
            "cc_bilateral_available",
            "mlo_bilateral_available",
            "bilateral_available",
            "complete_bilateral",
            "bilateral_pair_count",
        ]
    ].copy()

    audit["missing_bilateral_reason"] = ""

    audit.loc[
        (audit["cc_bilateral_available"] == 0)
        & (audit["mlo_bilateral_available"] == 0),
        "missing_bilateral_reason",
    ] = "missing_cc_and_mlo_bilateral_pairs"

    audit.loc[
        (audit["cc_bilateral_available"] == 0)
        & (audit["mlo_bilateral_available"] == 1),
        "missing_bilateral_reason",
    ] = "missing_cc_bilateral_pair"

    audit.loc[
        (audit["cc_bilateral_available"] == 1)
        & (audit["mlo_bilateral_available"] == 0),
        "missing_bilateral_reason",
    ] = "missing_mlo_bilateral_pair"

    audit.loc[
        audit["complete_bilateral"] == 1,
        "missing_bilateral_reason",
    ] = "complete_bilateral_alignment"

    return audit
