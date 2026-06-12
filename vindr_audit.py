```python
# src/organizational_sacu_mammography/metadata/vindr_audit.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import pandas as pd


@dataclass
class VinDrAuditReport:
    n_rows: int
    n_patients: int
    n_studies: int
    n_images: int
    missing_patient_id: int
    missing_study_id: int
    missing_laterality: int
    missing_view_position: int
    missing_label: int
    n_left: int
    n_right: int
    n_cc: int
    n_mlo: int
    n_positive: int
    n_negative: int

    def to_dict(self) -> Dict:
        return asdict(self)


def audit_vindr_metadata(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    study_col: str = "study_id",
    image_col: str = "image_id",
    laterality_col: str = "laterality",
    view_col: str = "view_position",
    label_col: str = "diagnosis_label",
) -> VinDrAuditReport:
    required = [
        patient_col,
        study_col,
        image_col,
        laterality_col,
        view_col,
        label_col,
    ]

    missing_cols = [c for c in required if c not in df.columns]

    if missing_cols:
        raise KeyError(f"Missing required VinDr audit columns: {missing_cols}")

    laterality = df[laterality_col].astype(str).str.upper()
    views = df[view_col].astype(str).str.upper()
    labels = df[label_col]

    return VinDrAuditReport(
        n_rows=int(len(df)),
        n_patients=int(df[patient_col].nunique(dropna=True)),
        n_studies=int(df[study_col].nunique(dropna=True)),
        n_images=int(df[image_col].nunique(dropna=True)),
        missing_patient_id=int(df[patient_col].isna().sum()),
        missing_study_id=int(df[study_col].isna().sum()),
        missing_laterality=int(df[laterality_col].isna().sum()),
        missing_view_position=int(df[view_col].isna().sum()),
        missing_label=int(df[label_col].isna().sum()),
        n_left=int((laterality == "LEFT").sum()),
        n_right=int((laterality == "RIGHT").sum()),
        n_cc=int((views == "CC").sum()),
        n_mlo=int((views == "MLO").sum()),
        n_positive=int((labels == 1).sum()),
        n_negative=int((labels == 0).sum()),
    )


def audit_vindr_to_dataframe(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    report = audit_vindr_metadata(df, **kwargs)
    return pd.DataFrame([report.to_dict()])
```
