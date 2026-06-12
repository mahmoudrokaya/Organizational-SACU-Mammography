```python
# src/organizational_sacu_mammography/metadata/patient_id_mapping.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

import pandas as pd


@dataclass
class PatientIDMapper:
    """
    Deterministic patient identifier mapper.

    The mapper converts original patient identifiers into stable internal IDs.
    It does not perform cryptographic anonymization; it only creates reproducible
    study identifiers for internal processing.
    """

    prefix: str = "P"
    width: int = 6
    mapping_: Dict[str, str] = field(default_factory=dict)

    def fit(self, patient_ids: Iterable) -> "PatientIDMapper":
        unique_ids = sorted({str(x).strip() for x in patient_ids if pd.notna(x)})

        self.mapping_ = {
            pid: f"{self.prefix}{i + 1:0{self.width}d}"
            for i, pid in enumerate(unique_ids)
        }

        return self

    def transform(self, patient_ids: Iterable) -> list[str]:
        if not self.mapping_:
            raise RuntimeError("PatientIDMapper must be fitted before transform().")

        mapped = []

        for x in patient_ids:
            key = str(x).strip()
            if key not in self.mapping_:
                raise KeyError(f"Unknown patient identifier: {key}")
            mapped.append(self.mapping_[key])

        return mapped

    def fit_transform(self, patient_ids: Iterable) -> list[str]:
        self.fit(patient_ids)
        return self.transform(patient_ids)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"original_patient_id": k, "mapped_patient_id": v}
                for k, v in self.mapping_.items()
            ]
        )


def create_patient_id_map(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    output_col: str = "mapped_patient_id",
    prefix: str = "P",
    width: int = 6,
) -> tuple[pd.DataFrame, PatientIDMapper]:
    if patient_col not in df.columns:
        raise KeyError(f"Missing patient identifier column: {patient_col}")

    mapper = PatientIDMapper(prefix=prefix, width=width)
    mapped = mapper.fit_transform(df[patient_col])

    out = df.copy()
    out[output_col] = mapped

    return out, mapper
```
