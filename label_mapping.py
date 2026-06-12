```python
# src/organizational_sacu_mammography/metadata/label_mapping.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Any

import pandas as pd


DEFAULT_BINARY_LABEL_MAP = {
    "BENIGN": 0,
    "B": 0,
    "NORMAL": 0,
    "NEGATIVE": 0,
    "0": 0,
    0: 0,
    "MALIGNANT": 1,
    "M": 1,
    "CANCER": 1,
    "POSITIVE": 1,
    "1": 1,
    1: 1,
}


@dataclass
class LabelMapper:
    """
    Diagnostic label mapper for binary mammography classification.
    """

    label_map: Dict[Any, int] = field(default_factory=lambda: DEFAULT_BINARY_LABEL_MAP.copy())
    unknown_policy: str = "raise"

    def _normalize_key(self, value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, str):
            return value.strip().upper()
        return value

    def map_one(self, value: Any) -> int | None:
        key = self._normalize_key(value)

        if key in self.label_map:
            return int(self.label_map[key])

        if self.unknown_policy == "ignore":
            return None

        if self.unknown_policy == "negative":
            return 0

        raise ValueError(f"Unknown diagnostic label: {value}")

    def transform(self, labels: Iterable) -> list[int | None]:
        return [self.map_one(x) for x in labels]


def map_binary_labels(
    df: pd.DataFrame,
    label_col: str,
    output_col: str = "diagnosis_label",
    unknown_policy: str = "raise",
) -> pd.DataFrame:
    if label_col not in df.columns:
        raise KeyError(f"Missing label column: {label_col}")

    mapper = LabelMapper(unknown_policy=unknown_policy)

    out = df.copy()
    out[output_col] = mapper.transform(out[label_col])

    if unknown_policy != "ignore":
        out[output_col] = out[output_col].astype(int)

    return out
```

```python
# src/organizational_sacu_mammography/metadata/metadata_harmonization.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from .label_mapping import map_binary_labels
from .patient_id_mapping import create_patient_id_map


@dataclass
class MetadataHarmonizer:
    """
    Harmonize mammography metadata into a common schema.
    """

    patient_col: str = "patient_id"
    study_col: str = "study_id"
    image_col: str = "image_id"
    date_col: str = "exam_date"
    laterality_col: str = "laterality"
    view_col: str = "view_position"
    label_col: str = "label"

    output_patient_col: str = "patient_id"
    output_study_col: str = "study_id"
    output_image_col: str = "image_id"
    output_date_col: str = "exam_date"
    output_laterality_col: str = "laterality"
    output_view_col: str = "view_position"
    output_label_col: str = "diagnosis_label"

    map_patient_ids: bool = False

    def _require_columns(self, df: pd.DataFrame) -> None:
        required = [
            self.patient_col,
            self.study_col,
            self.image_col,
            self.date_col,
            self.laterality_col,
            self.view_col,
            self.label_col,
        ]

        missing = [c for c in required if c not in df.columns]

        if missing:
            raise KeyError(f"Missing required metadata columns: {missing}")

    @staticmethod
    def _normalize_laterality(value) -> str:
        if pd.isna(value):
            return "UNKNOWN"

        v = str(value).strip().upper()

        if v in {"L", "LEFT"}:
            return "LEFT"

        if v in {"R", "RIGHT"}:
            return "RIGHT"

        return "UNKNOWN"

    @staticmethod
    def _normalize_view(value) -> str:
        if pd.isna(value):
            return "UNKNOWN"

        v = str(value).strip().upper().replace(" ", "").replace("-", "")

        if v in {"CC", "CRANIOCAUDAL"}:
            return "CC"

        if v in {"MLO", "MEDIOLATERALOBLIQUE", "MEDIOLATERALOBLIQUE"}:
            return "MLO"

        return "UNKNOWN"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_columns(df)

        out = pd.DataFrame()

        out[self.output_patient_col] = df[self.patient_col].astype(str).str.strip()
        out[self.output_study_col] = df[self.study_col].astype(str).str.strip()
        out[self.output_image_col] = df[self.image_col].astype(str).str.strip()

        out[self.output_date_col] = pd.to_datetime(
            df[self.date_col],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

        out[self.output_laterality_col] = df[self.laterality_col].map(self._normalize_laterality)
        out[self.output_view_col] = df[self.view_col].map(self._normalize_view)

        label_df = map_binary_labels(
            df=df,
            label_col=self.label_col,
            output_col=self.output_label_col,
            unknown_policy="raise",
        )

        out[self.output_label_col] = label_df[self.output_label_col].astype(int)

        if self.map_patient_ids:
            out, _ = create_patient_id_map(
                out,
                patient_col=self.output_patient_col,
                output_col=self.output_patient_col,
            )

        return out

    def audit(self, harmonized_df: pd.DataFrame) -> pd.DataFrame:
        records = []

        for col in [
            self.output_patient_col,
            self.output_study_col,
            self.output_image_col,
            self.output_date_col,
            self.output_laterality_col,
            self.output_view_col,
            self.output_label_col,
        ]:
            records.append(
                {
                    "field": col,
                    "missing_count": int(harmonized_df[col].isna().sum()),
                    "unique_count": int(harmonized_df[col].nunique(dropna=True)),
                }
            )

        records.append(
            {
                "field": "invalid_laterality",
                "missing_count": int((harmonized_df[self.output_laterality_col] == "UNKNOWN").sum()),
                "unique_count": 1,
            }
        )

        records.append(
            {
                "field": "invalid_view_position",
                "missing_count": int((harmonized_df[self.output_view_col] == "UNKNOWN").sum()),
                "unique_count": 1,
            }
        )

        return pd.DataFrame(records)


def harmonize_metadata(
    df: pd.DataFrame,
    column_map: Optional[Dict[str, str]] = None,
    map_patient_ids: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    column_map = column_map or {}

    harmonizer = MetadataHarmonizer(
        patient_col=column_map.get("patient_id", "patient_id"),
        study_col=column_map.get("study_id", "study_id"),
        image_col=column_map.get("image_id", "image_id"),
        date_col=column_map.get("exam_date", "exam_date"),
        laterality_col=column_map.get("laterality", "laterality"),
        view_col=column_map.get("view_position", "view_position"),
        label_col=column_map.get("label", "label"),
        map_patient_ids=map_patient_ids,
    )

    harmonized = harmonizer.transform(df)
    audit = harmonizer.audit(harmonized)

    return harmonized, audit
```
