# src/organizational_sacu_mammography/features/bilateral_features.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class BilateralFeatureConfig:
    """
    Configuration for bilateral left-right mammography feature extraction.

    The module compares feature dictionaries extracted from anatomically
    corresponding left and right breast views.
    """

    eps: float = 1e-8

    include_absolute_difference: bool = True
    include_signed_difference: bool = True
    include_ratio_features: bool = True
    include_similarity_features: bool = True


def _validate_pair(
    left_features: Dict[str, float],
    right_features: Dict[str, float],
) -> None:
    if not isinstance(left_features, dict):
        raise TypeError("left_features must be a dictionary.")

    if not isinstance(right_features, dict):
        raise TypeError("right_features must be a dictionary.")


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


def extract_bilateral_features(
    left_features: Dict[str, float],
    right_features: Dict[str, float],
    prefix: str = "bilateral",
    config: BilateralFeatureConfig = BilateralFeatureConfig(),
) -> Dict[str, float]:
    """
    Extract bilateral asymmetry descriptors from paired left-right views.
    """

    _validate_pair(left_features, right_features)

    keys = _shared_numeric_keys(left_features, right_features)

    if not keys:
        return {
            f"{prefix}_available": 0.0,
            f"{prefix}_n_shared_features": 0.0,
        }

    left_vec = np.asarray(
        [float(left_features[k]) for k in keys],
        dtype=np.float32,
    )

    right_vec = np.asarray(
        [float(right_features[k]) for k in keys],
        dtype=np.float32,
    )

    signed_diff = left_vec - right_vec
    abs_diff = np.abs(signed_diff)

    features: Dict[str, float] = {
        f"{prefix}_available": 1.0,
        f"{prefix}_n_shared_features": float(len(keys)),
        f"{prefix}_mean_absolute_asymmetry": float(np.mean(abs_diff)),
        f"{prefix}_max_absolute_asymmetry": float(np.max(abs_diff)),
        f"{prefix}_std_absolute_asymmetry": float(np.std(abs_diff)),
    }

    if config.include_signed_difference:
        features[f"{prefix}_mean_signed_difference"] = float(np.mean(signed_diff))
        features[f"{prefix}_std_signed_difference"] = float(np.std(signed_diff))

        for key, value in zip(keys, signed_diff):
            features[f"{prefix}_signeddiff_{key}"] = float(value)

    if config.include_absolute_difference:
        for key, value in zip(keys, abs_diff):
            features[f"{prefix}_absdiff_{key}"] = float(value)

    if config.include_ratio_features:
        ratio = left_vec / (np.abs(right_vec) + config.eps)

        features[f"{prefix}_mean_ratio"] = float(np.mean(ratio))
        features[f"{prefix}_std_ratio"] = float(np.std(ratio))

    if config.include_similarity_features:
        features[f"{prefix}_cosine_similarity"] = _cosine_similarity(
            left_vec,
            right_vec,
            config.eps,
        )

        if len(keys) > 1:
            corr = np.corrcoef(left_vec, right_vec)[0, 1]
            features[f"{prefix}_pearson_correlation"] = (
                float(corr)
                if np.isfinite(corr)
                else 0.0
            )
        else:
            features[f"{prefix}_pearson_correlation"] = 0.0

    return features


def extract_exam_bilateral_features(
    left_cc_features: Dict[str, float] | None = None,
    right_cc_features: Dict[str, float] | None = None,
    left_mlo_features: Dict[str, float] | None = None,
    right_mlo_features: Dict[str, float] | None = None,
    config: BilateralFeatureConfig = BilateralFeatureConfig(),
) -> Dict[str, float]:
    """
    Extract bilateral features for CC and MLO anatomical comparisons.
    """

    features: Dict[str, float] = {}

    if left_cc_features is not None and right_cc_features is not None:
        features.update(
            extract_bilateral_features(
                left_cc_features,
                right_cc_features,
                prefix="bilateral_cc",
                config=config,
            )
        )
    else:
        features["bilateral_cc_available"] = 0.0

    if left_mlo_features is not None and right_mlo_features is not None:
        features.update(
            extract_bilateral_features(
                left_mlo_features,
                right_mlo_features,
                prefix="bilateral_mlo",
                config=config,
            )
        )
    else:
        features["bilateral_mlo_available"] = 0.0

    features["bilateral_pair_count"] = float(
        features.get("bilateral_cc_available", 0.0)
        + features.get("bilateral_mlo_available", 0.0)
    )

    return features


def summarize_bilateral_features(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize extracted bilateral features.
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


def count_bilateral_features(
    n_base_features: int,
    config: BilateralFeatureConfig = BilateralFeatureConfig(),
) -> int:
    """
    Estimate bilateral feature dimensionality per left-right view pair.
    """

    count = 5

    if config.include_signed_difference:
        count += 2 + n_base_features

    if config.include_absolute_difference:
        count += n_base_features

    if config.include_ratio_features:
        count += 2

    if config.include_similarity_features:
        count += 2

    return count
