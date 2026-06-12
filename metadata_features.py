# src/organizational_sacu_mammography/features/metadata_features.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass
class MetadataFeatureConfig:
    """
    Configuration for structured metadata feature extraction.

    Metadata features are intentionally lightweight because the main SACU
    framework is driven by imaging-derived organizational pathways.
    """

    numeric_columns: tuple[str, ...] = (
        "age",
    )

    categorical_columns: tuple[str, ...] = (
        "breast_density",
        "scanner_manufacturer",
        "acquisition_site",
    )

    include_missing_indicators: bool = True
    one_hot_encode: bool = True
    prefix: str = "metadata"


def extract_metadata_features(
    metadata_row: pd.Series | Dict,
    config: MetadataFeatureConfig = MetadataFeatureConfig(),
) -> Dict[str, float]:
    """
    Extract metadata features from one examination-level metadata record.
    """

    if isinstance(metadata_row, dict):
        row = metadata_row
    else:
        row = metadata_row.to_dict()

    features: Dict[str, float] = {}

    for col in config.numeric_columns:
        value = row.get(col, np.nan)
        missing = pd.isna(value)

        features[f"{config.prefix}_{col}"] = (
            0.0 if missing else float(value)
        )

        if config.include_missing_indicators:
            features[f"{config.prefix}_{col}_missing"] = float(missing)

    for col in config.categorical_columns:
        value = row.get(col, None)
        missing = pd.isna(value) or str(value).strip() == ""

        if config.include_missing_indicators:
            features[f"{config.prefix}_{col}_missing"] = float(missing)

        if config.one_hot_encode:
            if not missing:
                clean_value = (
                    str(value)
                    .strip()
                    .lower()
                    .replace(" ", "_")
                    .replace("-", "_")
                )
                features[f"{config.prefix}_{col}_{clean_value}"] = 1.0
        else:
            features[f"{config.prefix}_{col}"] = 0.0 if missing else hash(str(value)) % 10000

    return features


def extract_metadata_feature_table(
    metadata_df: pd.DataFrame,
    config: MetadataFeatureConfig = MetadataFeatureConfig(),
) -> pd.DataFrame:
    """
    Extract metadata features for all rows in a dataframe.

    Because one-hot categorical columns may vary by row, the output is
    automatically filled with zeros for absent categories.
    """

    rows: List[Dict[str, float]] = []

    for _, row in metadata_df.iterrows():
        rows.append(
            extract_metadata_features(
                row,
                config=config,
            )
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.fillna(0.0)

    return out


def summarize_metadata_features(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize extracted metadata features.
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


def metadata_completeness_score(
    metadata_row: pd.Series | Dict,
    required_columns: Iterable[str],
) -> float:
    """
    Compute fraction of required metadata fields that are available.
    """

    if isinstance(metadata_row, dict):
        row = metadata_row
    else:
        row = metadata_row.to_dict()

    required = list(required_columns)

    if not required:
        return 1.0

    available = 0

    for col in required:
        value = row.get(col, None)
        if pd.notna(value) and str(value).strip() != "":
            available += 1

    return float(available / len(required))
