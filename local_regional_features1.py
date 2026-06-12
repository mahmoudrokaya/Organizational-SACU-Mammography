# src/organizational_sacu_mammography/features/local_regional_features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class LocalRegionalFeatureConfig:
    """
    Configuration for local-regional mammography feature extraction.
    """

    eps: float = 1e-8
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    grid_rows: int = 3
    grid_cols: int = 3
    include_global_features: bool = True
    include_gradient_features: bool = True
    include_regional_features: bool = True


def _validate_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray.")

    if image.ndim != 2:
        raise ValueError("image must be a 2D grayscale mammography image.")

    if image.size == 0:
        raise ValueError("image is empty.")

    return image.astype(np.float32, copy=False)


def _normalize_image(
    image: np.ndarray,
    config: LocalRegionalFeatureConfig,
) -> np.ndarray:
    low = np.percentile(image, config.lower_percentile)
    high = np.percentile(image, config.upper_percentile)

    if high - low < config.eps:
        return np.zeros_like(image, dtype=np.float32)

    clipped = np.clip(image, low, high)
    return ((clipped - low) / (high - low + config.eps)).astype(np.float32)


def _basic_statistics(
    values: np.ndarray,
    prefix: str,
    eps: float,
) -> Dict[str, float]:
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_iqr": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p90": 0.0,
        }

    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_iqr": float(p75 - p25),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def _gradient_features(
    image: np.ndarray,
    prefix: str,
    config: LocalRegionalFeatureConfig,
) -> Dict[str, float]:
    gy, gx = np.gradient(image)
    magnitude = np.sqrt(gx**2 + gy**2)

    features = _basic_statistics(
        magnitude.ravel(),
        f"{prefix}_gradient",
        config.eps,
    )

    features[f"{prefix}_edge_density"] = float(
        np.mean(magnitude > np.percentile(magnitude, 75))
    )

    return features


def _regional_features(
    image: np.ndarray,
    prefix: str,
    config: LocalRegionalFeatureConfig,
) -> Dict[str, float]:
    h, w = image.shape
    row_edges = np.linspace(0, h, config.grid_rows + 1, dtype=int)
    col_edges = np.linspace(0, w, config.grid_cols + 1, dtype=int)

    records: Dict[str, float] = {}
    regional_means: List[float] = []

    for r in range(config.grid_rows):
        for c in range(config.grid_cols):
            patch = image[
                row_edges[r]:row_edges[r + 1],
                col_edges[c]:col_edges[c + 1],
            ]

            patch_prefix = f"{prefix}_region_r{r}_c{c}"
            stats = _basic_statistics(
                patch.ravel(),
                patch_prefix,
                config.eps,
            )

            records.update(stats)
            regional_means.append(stats[f"{patch_prefix}_mean"])

    regional_means_arr = np.asarray(regional_means, dtype=np.float32)

    records[f"{prefix}_regional_mean_range"] = float(
        np.max(regional_means_arr) - np.min(regional_means_arr)
    )

    records[f"{prefix}_regional_mean_std"] = float(
        np.std(regional_means_arr)
    )

    return records


def extract_local_regional_features(
    image: np.ndarray,
    view_prefix: str,
    config: LocalRegionalFeatureConfig = LocalRegionalFeatureConfig(),
) -> Dict[str, float]:
    """
    Extract local-regional shallow mammography descriptors from one view.
    """

    image = _validate_image(image)
    image = _normalize_image(image, config)

    features: Dict[str, float] = {}

    if config.include_global_features:
        features.update(
            _basic_statistics(
                image.ravel(),
                f"{view_prefix}_intensity",
                config.eps,
            )
        )

        features[f"{view_prefix}_nonzero_fraction"] = float(
            np.mean(image > config.eps)
        )

        features[f"{view_prefix}_high_density_fraction"] = float(
            np.mean(image > np.percentile(image, 75))
        )

    if config.include_gradient_features:
        features.update(
            _gradient_features(
                image,
                view_prefix,
                config,
            )
        )

    if config.include_regional_features:
        features.update(
            _regional_features(
                image,
                view_prefix,
                config,
            )
        )

    return features


def extract_local_regional_feature_table(
    image_records: List[Tuple[str, str, np.ndarray]],
    config: LocalRegionalFeatureConfig = LocalRegionalFeatureConfig(),
) -> pd.DataFrame:
    """
    Build a feature table from image records.

    Each record must contain:
    (record_id, view_prefix, image_array)
    """

    rows: List[Dict[str, float | str]] = []

    for record_id, view_prefix, image in image_records:
        features = extract_local_regional_features(
            image=image,
            view_prefix=view_prefix,
            config=config,
        )

        row: Dict[str, float | str] = {
            "record_id": record_id,
            "view_prefix": view_prefix,
        }

        row.update(features)
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_local_regional_features(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize extracted local-regional feature completeness.
    """

    if feature_df.empty:
        return pd.DataFrame(
            columns=["metric", "value"]
        )

    numeric_cols = feature_df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    summary = [
        {
            "metric": "n_records",
            "value": int(len(feature_df)),
        },
        {
            "metric": "n_numeric_features",
            "value": int(len(numeric_cols)),
        },
        {
            "metric": "missing_numeric_values",
            "value": int(feature_df[numeric_cols].isna().sum().sum()),
        },
        {
            "metric": "infinite_numeric_values",
            "value": int(
                np.isinf(feature_df[numeric_cols].to_numpy()).sum()
            ),
        },
    ]

    return pd.DataFrame(summary)


def validate_local_regional_feature_table(
    feature_df: pd.DataFrame,
) -> None:
    """
    Validate that the local-regional feature table is modeling-ready.
    """

    if feature_df.empty:
        raise ValueError("Local-regional feature table is empty.")

    numeric_df = feature_df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        raise ValueError("No numeric local-regional features were extracted.")

    if numeric_df.isna().any().any():
        raise ValueError("Local-regional feature table contains NaN values.")

    if np.isinf(numeric_df.to_numpy()).any():
        raise ValueError("Local-regional feature table contains infinite values.")
