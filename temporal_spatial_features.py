# src/organizational_sacu_mammography/features/temporal_spatial_features.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class TemporalSpatialFeatureConfig:
    """
    Configuration for temporal-spatial feature extraction.

    The module compares feature dictionaries extracted from a current
    mammography examination and its prior examination.
    """

    eps: float = 1e-8

    include_absolute_change: bool = True
    include_signed_change: bool = True
    include_ratio_change: bool = True
    include_similarity_features: bool = True
    include_temporal_gap: bool = True


def _validate_pair(
    current_features: Dict[str, float],
    prior_features: Dict[str, float],
) -> None:
    if not isinstance(current_features, dict):
        raise TypeError("current_features must be a dictionary.")

    if not isinstance(prior_features, dict):
        raise TypeError("prior_features must be a dictionary.")


def _shared_numeric_keys(
    a: Dict[str, float],
    b: Dict[str, float],
) -> List[str]:
    shared = sorted(set(a.keys()).intersection(set(b.keys())))

    numeric = []

    for key in shared:
        try:
            float(a[key])
            float(b[key])
            numeric.append(key)
        except Exception:
            continue

    return numeric


def _cosine_similarity(
    x: np.ndarray,
    y: np.ndarray,
    eps: float,
) -> float:
    denom = np.linalg.norm(x) * np.linalg.norm(y)

    if denom < eps:
        return 0.0

    return float(np.dot(x, y) / denom)


def extract_temporal_spatial_features(
    current_features: Dict[str, float],
    prior_features: Dict[str, float],
    temporal_gap_days: float | None = None,
    prefix: str = "temporal",
    config: TemporalSpatialFeatureConfig = TemporalSpatialFeatureConfig(),
) -> Dict[str, float]:
    """
    Extract longitudinal temporal-spatial change descriptors.

    Parameters
    ----------
    current_features:
        Feature dictionary from the current examination.

    prior_features:
        Feature dictionary from the previous examination.

    temporal_gap_days:
        Time difference between current and prior examination in days.

    prefix:
        Output feature prefix.
    """

    _validate_pair(current_features, prior_features)

    keys = _shared_numeric_keys(current_features, prior_features)

    if not keys:
        features = {
            f"{prefix}_available": 0.0,
            f"{prefix}_n_shared_features": 0.0,
        }

        if config.include_temporal_gap:
            features[f"{prefix}_gap_days"] = (
                float(temporal_gap_days)
                if temporal_gap_days is not None
                else 0.0
            )

        return features

    current_vec = np.asarray(
        [float(current_features[k]) for k in keys],
        dtype=np.float32,
    )

    prior_vec = np.asarray(
        [float(prior_features[k]) for k in keys],
        dtype=np.float32,
    )

    signed_change = current_vec - prior_vec
    abs_change = np.abs(signed_change)

    features: Dict[str, float] = {
        f"{prefix}_available": 1.0,
        f"{prefix}_n_shared_features": float(len(keys)),
        f"{prefix}_mean_absolute_change": float(np.mean(abs_change)),
        f"{prefix}_max_absolute_change": float(np.max(abs_change)),
        f"{prefix}_std_absolute_change": float(np.std(abs_change)),
    }

    if config.include_signed_change:
        features[f"{prefix}_mean_signed_change"] = float(np.mean(signed_change))
        features[f"{prefix}_std_signed_change"] = float(np.std(signed_change))

        for key, value in zip(keys, signed_change):
            features[f"{prefix}_signedchange_{key}"] = float(value)

    if config.include_absolute_change:
        for key, value in zip(keys, abs_change):
            features[f"{prefix}_abschange_{key}"] = float(value)

    if config.include_ratio_change:
        ratio = current_vec / (np.abs(prior_vec) + config.eps)

        features[f"{prefix}_mean_ratio_change"] = float(np.mean(ratio))
        features[f"{prefix}_std_ratio_change"] = float(np.std(ratio))

    if config.include_similarity_features:
        features[f"{prefix}_cosine_similarity"] = _cosine_similarity(
            current_vec,
            prior_vec,
            config.eps,
        )

        if len(keys) > 1:
            corr = np.corrcoef(current_vec, prior_vec)[0, 1]
            features[f"{prefix}_pearson_correlation"] = (
                float(corr)
                if np.isfinite(corr)
                else 0.0
            )
        else:
            features[f"{prefix}_pearson_correlation"] = 0.0

    if config.include_temporal_gap:
        features[f"{prefix}_gap_days"] = (
            float(temporal_gap_days)
            if temporal_gap_days is not None
            else 0.0
        )

        if temporal_gap_days is not None and temporal_gap_days > 0:
            features[f"{prefix}_mean_absolute_change_per_day"] = (
                features[f"{prefix}_mean_absolute_change"] / temporal_gap_days
            )
        else:
            features[f"{prefix}_mean_absolute_change_per_day"] = 0.0

    return features


def extract_exam_temporal_spatial_features(
    current_exam_features: Dict[str, float] | None,
    prior_exam_features: Dict[str, float] | None,
    temporal_gap_days: float | None = None,
    config: TemporalSpatialFeatureConfig = TemporalSpatialFeatureConfig(),
) -> Dict[str, float]:
    """
    Extract temporal-spatial descriptors for an examination.

    If prior examination features are unavailable, a masked temporal feature
    representation is returned.
    """

    if current_exam_features is None or prior_exam_features is None:
        return {
            "temporal_available": 0.0,
            "temporal_n_shared_features": 0.0,
            "temporal_gap_days": (
                float(temporal_gap_days)
                if temporal_gap_days is not None
                else 0.0
            ),
            "temporal_mean_absolute_change": 0.0,
            "temporal_max_absolute_change": 0.0,
            "temporal_std_absolute_change": 0.0,
            "temporal_mean_absolute_change_per_day": 0.0,
        }

    return extract_temporal_spatial_features(
        current_features=current_exam_features,
        prior_features=prior_exam_features,
        temporal_gap_days=temporal_gap_days,
        prefix="temporal",
        config=config,
    )


def summarize_temporal_spatial_features(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize extracted temporal-spatial features.
    """

    if feature_df.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "mean",
                "std",
                "min",
                "max",
            ]
        )

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns

    rows = []

    for col in numeric_cols:
        rows.append(
            {
                "feature": col,
                "mean": float(feature_df[col].mean()),
                "std": float(feature_df[col].std()),
                "min": float(feature_df[col].min()),
                "max": float(feature_df[col].max()),
            }
        )

    return pd.DataFrame(rows)


def count_temporal_spatial_features(
    n_base_features: int,
    config: TemporalSpatialFeatureConfig = TemporalSpatialFeatureConfig(),
) -> int:
    """
    Estimate temporal-spatial feature dimensionality per current-prior pair.
    """

    count = 5

    if config.include_signed_change:
        count += 2 + n_base_features

    if config.include_absolute_change:
        count += n_base_features

    if config.include_ratio_change:
        count += 2

    if config.include_similarity_features:
        count += 2

    if config.include_temporal_gap:
        count += 2

    return count
