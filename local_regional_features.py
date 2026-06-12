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

    Images are partitioned into a regular grid and statistics are
    computed for each region.
    """

    grid_rows: int = 4
    grid_cols: int = 4

    include_global_features: bool = True

    eps: float = 1e-8


def _validate_image(image: np.ndarray) -> None:
    if image is None:
        raise ValueError("Image is None.")

    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D image, received shape {image.shape}"
        )


def _safe_skew(x: np.ndarray, eps: float) -> float:
    std = np.std(x)

    if std < eps:
        return 0.0

    return float(np.mean(((x - np.mean(x)) / std) ** 3))


def _safe_kurtosis(x: np.ndarray, eps: float) -> float:
    std = np.std(x)

    if std < eps:
        return 0.0

    return float(np.mean(((x - np.mean(x)) / std) ** 4))


def _entropy(x: np.ndarray) -> float:
    hist, _ = np.histogram(
        x.ravel(),
        bins=64,
        density=True,
    )

    hist = hist[hist > 0]

    if len(hist) == 0:
        return 0.0

    return float(-np.sum(hist * np.log2(hist)))


def _region_statistics(
    region: np.ndarray,
    prefix: str,
    eps: float,
) -> Dict[str, float]:

    region = region.astype(np.float32)

    return {
        f"{prefix}_mean": float(np.mean(region)),
        f"{prefix}_std": float(np.std(region)),
        f"{prefix}_median": float(np.median(region)),
        f"{prefix}_min": float(np.min(region)),
        f"{prefix}_max": float(np.max(region)),
        f"{prefix}_skewness": _safe_skew(region, eps),
        f"{prefix}_kurtosis": _safe_kurtosis(region, eps),
        f"{prefix}_entropy": _entropy(region),
    }


def _split_grid(
    image: np.ndarray,
    rows: int,
    cols: int,
) -> List[Tuple[int, int, np.ndarray]]:
    h, w = image.shape

    row_edges = np.linspace(
        0,
        h,
        rows + 1,
        dtype=int,
    )

    col_edges = np.linspace(
        0,
        w,
        cols + 1,
        dtype=int,
    )

    patches = []

    for r in range(rows):
        for c in range(cols):

            patch = image[
                row_edges[r]:row_edges[r + 1],
                col_edges[c]:col_edges[c + 1],
            ]

            patches.append((r, c, patch))

    return patches


def extract_local_regional_features(
    image: np.ndarray,
    config: LocalRegionalFeatureConfig = LocalRegionalFeatureConfig(),
) -> Dict[str, float]:
    """
    Extract local-regional tissue descriptors.

    Returns a flat feature dictionary suitable for Stage2A
    organizational feature extraction.
    """

    _validate_image(image)

    image = image.astype(np.float32)

    features: Dict[str, float] = {}

    if config.include_global_features:

        features.update(
            _region_statistics(
                image,
                "global",
                config.eps,
            )
        )

    patches = _split_grid(
        image,
        config.grid_rows,
        config.grid_cols,
    )

    for r, c, patch in patches:

        prefix = f"local_r{r}_c{c}"

        features.update(
            _region_statistics(
                patch,
                prefix,
                config.eps,
            )
        )

    return features


def extract_batch_local_regional_features(
    images: List[np.ndarray],
    config: LocalRegionalFeatureConfig = LocalRegionalFeatureConfig(),
) -> pd.DataFrame:
    """
    Extract local-regional features for multiple images.
    """

    rows = []

    for idx, image in enumerate(images):

        feat = extract_local_regional_features(
            image,
            config=config,
        )

        feat["sample_index"] = idx

        rows.append(feat)

    return pd.DataFrame(rows)


def summarize_local_regional_features(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize extracted local-regional features.
    """

    numeric_cols = feature_df.select_dtypes(
        include=[np.number]
    ).columns

    records = []

    for col in numeric_cols:

        records.append(
            {
                "feature": col,
                "mean": float(feature_df[col].mean()),
                "std": float(feature_df[col].std()),
                "min": float(feature_df[col].min()),
                "max": float(feature_df[col].max()),
            }
        )

    return pd.DataFrame(records)


def count_local_regional_features(
    config: LocalRegionalFeatureConfig = LocalRegionalFeatureConfig(),
) -> int:
    """
    Estimate feature dimensionality.

    Useful for Stage2A audit reports.
    """

    per_region = 8

    n_regions = (
        config.grid_rows
        * config.grid_cols
    )

    total = per_region * n_regions

    if config.include_global_features:
        total += per_region

    return total
