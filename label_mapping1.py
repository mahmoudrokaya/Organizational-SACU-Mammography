# src/organizational_sacu_mammography/metadata/label_mapping.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import pandas as pd


DEFAULT_BINARY_LABEL_MAP: Dict[Any, int] = {
    # Negative / benign labels
    "BENIGN": 0,
    "B": 0,
    "NORMAL": 0,
    "NEGATIVE": 0,
    "NO CANCER": 0,
    "NON-MALIGNANT": 0,
    "0": 0,
    0: 0,

    # Positive / malignant labels
    "MALIGNANT": 1,
    "M": 1,
    "CANCER": 1,
    "POSITIVE": 1,
    "ABNORMAL": 1,
    "1": 1,
    1: 1,
}


@dataclass
class LabelMapper:
    """
    Binary diagnostic label mapper.

    Converts heterogeneous mammography labels into a binary target:

    - 0 = benign / negative
    - 1 = malignant / positive
    """

    label_map: Dict[Any, int] = field(
        default_factory=lambda: DEFAULT_BINARY_LABEL_MAP.copy()
    )
    unknown_policy: str = "raise"

    def _normalize_key(self, value: Any) -> Any:
        if pd.isna(value):
            return None

        if isinstance(value, str):
            return value.strip().upper()

        return value

    def map_one(self, value: Any) -> Optional[int]:
        key = self._normalize_key(value)

        if key in self.label_map:
            return int(self.label_map[key])

        if self.unknown_policy == "ignore":
            return None

        if self.unknown_policy == "negative":
            return 0

        if self.unknown_policy == "positive":
            return 1

        raise ValueError(f"Unknown diagnostic label: {value}")

    def transform(self, labels: Iterable[Any]) -> list[Optional[int]]:
        return [self.map_one(label) for label in labels]

    def transform_series(self, series: pd.Series) -> pd.Series:
        mapped = self.transform(series)

        out = pd.Series(mapped, index=series.index, name=series.name)

        if self.unknown_policy != "ignore":
            out = out.astype(int)

        return out

    def inverse_transform(self, labels: Iterable[int]) -> list[str]:
        inverse = {
            0: "benign",
            1: "malignant",
        }

        result = []

        for label in labels:
            if label not in inverse:
                raise ValueError(f"Unknown binary label: {label}")
            result.append(inverse[int(label)])

        return result


def map_binary_labels(
    df: pd.DataFrame,
    label_col: str,
    output_col: str = "diagnosis_label",
    label_map: Optional[Dict[Any, int]] = None,
    unknown_policy: str = "raise",
    drop_unknown: bool = False,
) -> pd.DataFrame:
    """
    Map a dataframe label column into a binary diagnosis column.

    Parameters
    ----------
    df:
        Input dataframe.

    label_col:
        Name of the source label column.

    output_col:
        Name of the output binary label column.

    label_map:
        Optional custom label map. If None, DEFAULT_BINARY_LABEL_MAP is used.

    unknown_policy:
        Policy for unknown labels:
        - "raise": raise an error
        - "ignore": map unknown labels to None
        - "negative": map unknown labels to 0
        - "positive": map unknown labels to 1

    drop_unknown:
        If True, rows with unmapped labels are removed. This is only relevant
        when unknown_policy="ignore".
    """

    if label_col not in df.columns:
        raise KeyError(f"Missing label column: {label_col}")

    mapper = LabelMapper(
        label_map=label_map or DEFAULT_BINARY_LABEL_MAP.copy(),
        unknown_policy=unknown_policy,
    )

    out = df.copy()
    out[output_col] = mapper.transform_series(out[label_col])

    if drop_unknown:
        out = out.dropna(subset=[output_col]).copy()
        out[output_col] = out[output_col].astype(int)

    return out


def summarize_labels(
    df: pd.DataFrame,
    label_col: str = "diagnosis_label",
) -> pd.DataFrame:
    """
    Return a compact summary of binary label counts.
    """

    if label_col not in df.columns:
        raise KeyError(f"Missing label column: {label_col}")

    counts = (
        df[label_col]
        .value_counts(dropna=False)
        .rename_axis(label_col)
        .reset_index(name="count")
    )

    counts["fraction"] = counts["count"] / len(df)

    return counts
