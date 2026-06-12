# src/organizational_sacu_mammography/representation/pathway_masking.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class PathwayMaskConfig:
    """
    Configuration for constructing SACU pathway availability masks.

    These masks indicate whether each organizational diagnostic pathway
    has sufficient information for a given examination.
    """

    patient_col: str = "patient_id"
    study_col: str = "study_id"
    label_col: str = "diagnosis_label"

    # Multi-view availability columns
    left_cc_available_col: str = "left_cc_available"
    left_mlo_available_col: str = "left_mlo_available"
    right_cc_available_col: str = "right_cc_available"
    right_mlo_available_col: str = "right_mlo_available"

    # Bilateral availability columns
    cc_bilateral_available_col: str = "cc_bilateral_available"
    mlo_bilateral_available_col: str = "mlo_bilateral_available"

    # Temporal availability columns
    temporal_case_col: str = "temporal_case"
    has_prior_exam_col: str = "has_prior_exam"

    # Metadata availability
    metadata_required_cols: tuple[str, ...] = ()

    require_full_multiview_exam: bool = False
    require_any_bilateral_pair: bool = True
    require_prior_exam_for_temporal: bool = True


def _safe_col(
    df: pd.DataFrame,
    col: str,
    default: int = 0,
) -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default).astype(int)
    return pd.Series(default, index=df.index, dtype=int)


def build_pathway_masks(
    df: pd.DataFrame,
    config: PathwayMaskConfig = PathwayMaskConfig(),
) -> pd.DataFrame:
    """
    Build binary pathway availability masks.

    Output mask columns:
    - local_regional_mask
    - multiview_mask
    - bilateral_mask
    - temporal_spatial_mask
    - metadata_mask
    - adaptive_control_mask
    """

    out = df.copy()

    left_cc = _safe_col(out, config.left_cc_available_col)
    left_mlo = _safe_col(out, config.left_mlo_available_col)
    right_cc = _safe_col(out, config.right_cc_available_col)
    right_mlo = _safe_col(out, config.right_mlo_available_col)

    available_view_count = left_cc + left_mlo + right_cc + right_mlo

    # Local-regional pathway requires at least one valid mammography view.
    out["local_regional_mask"] = (available_view_count >= 1).astype(int)

    # Multi-view pathway can be either strict full-exam or at least one CC/MLO pair.
    left_pair = (left_cc == 1) & (left_mlo == 1)
    right_pair = (right_cc == 1) & (right_mlo == 1)

    if config.require_full_multiview_exam:
        out["multiview_mask"] = (
            (left_cc == 1)
            & (left_mlo == 1)
            & (right_cc == 1)
            & (right_mlo == 1)
        ).astype(int)
    else:
        out["multiview_mask"] = (left_pair | right_pair).astype(int)

    # Bilateral pathway requires either CC or MLO bilateral availability.
    cc_bilateral = _safe_col(out, config.cc_bilateral_available_col)
    mlo_bilateral = _safe_col(out, config.mlo_bilateral_available_col)

    if config.require_any_bilateral_pair:
        out["bilateral_mask"] = (
            (cc_bilateral == 1)
            | (mlo_bilateral == 1)
        ).astype(int)
    else:
        out["bilateral_mask"] = (
            (cc_bilateral == 1)
            & (mlo_bilateral == 1)
        ).astype(int)

    # Temporal-spatial pathway requires a prior exam if configured.
    temporal_case = _safe_col(out, config.temporal_case_col)
    has_prior = _safe_col(out, config.has_prior_exam_col)

    if config.require_prior_exam_for_temporal:
        out["temporal_spatial_mask"] = (
            (temporal_case == 1)
            & (has_prior == 1)
        ).astype(int)
    else:
        out["temporal_spatial_mask"] = (temporal_case == 1).astype(int)

    # Metadata pathway depends on required metadata fields.
    if config.metadata_required_cols:
        metadata_available = pd.Series(True, index=out.index)

        for col in config.metadata_required_cols:
            if col not in out.columns:
                metadata_available &= False
            else:
                metadata_available &= out[col].notna()

        out["metadata_mask"] = metadata_available.astype(int)
    else:
        out["metadata_mask"] = 1

    # Adaptive-control pathway is always available because it is derived
    # from pathway availability and examination completeness.
    out["adaptive_control_mask"] = 1

    out["active_pathway_count"] = out[
        [
            "local_regional_mask",
            "multiview_mask",
            "bilateral_mask",
            "temporal_spatial_mask",
            "metadata_mask",
            "adaptive_control_mask",
        ]
    ].sum(axis=1)

    out["pathway_missingness_count"] = 6 - out["active_pathway_count"]

    out["view_completeness_score"] = available_view_count / 4.0

    out["bilateral_availability_score"] = (
        cc_bilateral + mlo_bilateral
    ) / 2.0

    out["temporal_availability_score"] = out["temporal_spatial_mask"]

    out["metadata_completeness_score"] = out["metadata_mask"]

    out["examination_complexity_score"] = (
        out["view_completeness_score"]
        + out["bilateral_availability_score"]
        + out["temporal_availability_score"]
        + out["metadata_completeness_score"]
    ) / 4.0

    return out


def summarize_pathway_masks(
    masked_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize pathway availability across the cohort.
    """

    mask_cols = [
        "local_regional_mask",
        "multiview_mask",
        "bilateral_mask",
        "temporal_spatial_mask",
        "metadata_mask",
        "adaptive_control_mask",
    ]

    missing = [c for c in mask_cols if c not in masked_df.columns]

    if missing:
        raise KeyError(f"Missing pathway mask columns: {missing}")

    records: List[Dict] = []

    n = len(masked_df)

    for col in mask_cols:
        active_count = int(masked_df[col].sum())
        records.append(
            {
                "pathway": col.replace("_mask", ""),
                "active_count": active_count,
                "inactive_count": int(n - active_count),
                "active_fraction": float(active_count / n) if n else 0.0,
            }
        )

    records.append(
        {
            "pathway": "mean_active_pathway_count",
            "active_count": float(masked_df["active_pathway_count"].mean()),
            "inactive_count": None,
            "active_fraction": None,
        }
    )

    return pd.DataFrame(records)


def audit_pathway_masks(
    masked_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate examination-level pathway availability audit.
    """

    required_cols = [
        "patient_id",
        "study_id",
        "local_regional_mask",
        "multiview_mask",
        "bilateral_mask",
        "temporal_spatial_mask",
        "metadata_mask",
        "adaptive_control_mask",
        "active_pathway_count",
        "pathway_missingness_count",
    ]

    available_cols = [c for c in required_cols if c in masked_df.columns]

    return masked_df[available_cols].copy()
