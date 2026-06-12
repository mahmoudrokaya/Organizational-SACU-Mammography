```python
# src/organizational_sacu_mammography/preprocessing/dicom_loading.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pydicom


@dataclass
class DICOMImage:
    path: Path
    pixel_array: np.ndarray
    metadata: Dict[str, Any]


def load_dicom_metadata(path: str | Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"DICOM file not found: {path}")

    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)

    metadata = {
        "patient_id": getattr(ds, "PatientID", None),
        "study_instance_uid": getattr(ds, "StudyInstanceUID", None),
        "series_instance_uid": getattr(ds, "SeriesInstanceUID", None),
        "sop_instance_uid": getattr(ds, "SOPInstanceUID", None),
        "study_date": getattr(ds, "StudyDate", None),
        "view_position": getattr(ds, "ViewPosition", None),
        "image_laterality": getattr(ds, "ImageLaterality", None),
        "laterality": getattr(ds, "Laterality", None),
        "photometric_interpretation": getattr(ds, "PhotometricInterpretation", None),
        "rows": getattr(ds, "Rows", None),
        "columns": getattr(ds, "Columns", None),
        "manufacturer": getattr(ds, "Manufacturer", None),
        "modality": getattr(ds, "Modality", None),
    }

    return metadata


def load_dicom_image(path: str | Path, apply_rescale: bool = True) -> DICOMImage:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"DICOM file not found: {path}")

    ds = pydicom.dcmread(str(path), force=True)

    if not hasattr(ds, "pixel_array"):
        raise ValueError(f"DICOM file has no pixel data: {path}")

    image = ds.pixel_array.astype(np.float32)

    if apply_rescale:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        image = image * slope + intercept

    metadata = load_dicom_metadata(path)

    return DICOMImage(
        path=path,
        pixel_array=image,
        metadata=metadata,
    )
```
