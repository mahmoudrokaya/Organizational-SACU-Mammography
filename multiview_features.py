# src/organizational_sacu_mammography/features/multiview_features.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class MultiViewFeatureConfig:
    """
    Configuration for multi-view CC/MLO feature extraction.

    The module compares paired mammography views from the same breast.
    """

    eps: float = 1e-8

    include_absolute_difference: bool = True
    include_ratio_features: bool = True
    include_similarity_features: bool = True


def _validate_pair(
    cc_features: Dict[str, float],
    mlo_features: Dict[str, float],
) -> None:
    if not isinstance(cc_features, dict):
        raise TypeError("cc_features must be a dictionary.")

    if not isinstance(mlo_features, dict):
        raise TypeError("mlo_features must be a dictionary.")


def _shared_numeric_keys(
    a: Dict[str, float],
    b: Dict[str, float],
) -> List[str]:
    shared = sorted(set(a.keys()).intersection(set(b.keys())))

    numeric = []

    for key in shared:
        if np.isscalar(a[key]) and np.isscalar(b[key]):
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


def extract_multiview_features(
    cc_features: Dict[str, float],
    mlo_features: Dict[str, float],
    prefix: str = "multiview",
    config: MultiViewFeatureConfig = MultiViewFeatureConfig(),
) -> Dict[str, float]:
    """
    Extract paired CC/MLO comparison features.

    Parameters
    ----------
    cc_features:
        Feature dictionary extracted from the CC image.

    mlo_features:
        Feature dictionary extracted from the MLO image.

    prefix:
        Output feature prefix.

    Returns
    -------
    Dict[str, float]
        Multi-view feature dictionary.
    """

    _validate_pair(cc_features, mlo_features)

    keys = _shared_numeric_keys(
        cc_features,
        mlo_features,
    )

    if not keys:
        return {
            f"{prefix}_available": 0.0,
            f"{prefix}_n_shared_features": 0.0,
        }

    cc_vec = np.asarray(
        [float(cc_features[k]) for k in keys],
        dtype=np.float32,
    )

    mlo_vec = np.asarray(
        [float(mlo_features[k]) for k in keys],
        dtype=np.float32,
    )

    diff = cc_vec - mlo_vec
    abs_diff = np.abs(diff)

    features: Dict[str, float] = {
        f"{prefix}_available": 1.0,
        f"{prefix}_n_shared_features": float(len(keys)),
        f"{prefix}_mean_difference": float(np.mean(diff)),
        f"{prefix}_std_difference": float(np.std(diff)),
        f"{prefix}_mean_absolute_difference": float(np.mean(abs_diff)),
        f"{prefix}_max_absolute_difference": float(np.max(abs_diff)),
    }

    if config.include_absolute_difference:
        for key, value in zip(keys, abs_diff):
            features[f"{prefix}_absdiff_{key}"] = float(value)

    if config.include_ratio_features:
        ratio = cc_vec / (np.abs(mlo_vec) + config.eps)

        features[f"{prefix}_mean_ratio"] = float(np.mean(ratio))
        features[f"{prefix}_std_ratio"] = float(np.std(ratio))

    if config.include_similarity_features:
        features[f"{prefix}_cosine_similarity"] = _cosine_similarity(
            cc_vec,
            mlo_vec,
            config.eps,
        )

        if len(keys) > 1:
            corr = np.corrcoef(cc_vec, mlo_vec)[0, 1]
            features[f"{prefix}_pearson_correlation"] = (
                float(corr)
                if np.isfinite(corr)
                else 0.0
            )
        else:
            features[f"{prefix}_pearson_correlation"] = 0.0

    return features


def extract_breast_multiview_features(
    left_cc_features: Dict[str, float] | None = None,
    left_mlo_features: Dict[str, float] | None = None,
    right_cc_features: Dict[str, float] | None = None,
    right_mlo_features: Dict[str, float] | None = None,
    config: MultiViewFeatureConfig = MultiViewFeatureConfig(),
) -> Dict[str, float]:
    """
    Extract multi-view features for left and right breast when available.
    """

    features: Dict[str, float] = {}

    if left_cc_features is not None and left_mlo_features is not None:
        features.update(
            extract_multiview_features(
                left_cc_features,
                left_mlo_features,
                prefix="multiview_left_cc_mlo",
                config=config,
            )
        )
    else:
        features["multiview_left_cc_mlo_available"] = 0.0

    if right_cc_features is not None and right_mlo_features is not None:
        features.update(
            extract_multiview_features(
                right_cc_features,
                right_mlo_features,
                prefix="multiview_right_cc_mlo",
                config=config,
            )
        )
    else:
        features["multiview_right_cc_mlo_available"] = 0.0

    features["multiview_pair_count"] = float(
        features.get("multiview_left_cc_mlo_available", 0.0)
        + features.get("multiview_right_cc_mlo_available", 0.0)
    )

    return features


def summarize_multiview_features(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize extracted multi-view features.
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

    numeric_cols = feature_df.select_dtypes(
        include=[np.number]
    ).columns

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


def count_multiview_features(
    n_base_features: int,
    config: MultiViewFeatureConfig = MultiViewFeatureConfig(),
) -> int:
    """
    Estimate the number of multi-view features per paired CC/MLO comparison.
    """

    count = 6

    if config.include_absolute_difference:
        count += n_base_features

    if config.include_ratio_features:
        count += 2

    if config.include_similarity_features:
        count += 2

    return count
