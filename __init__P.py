```python
# src/organizational_sacu_mammography/preprocessing/__init__.py

"""
Preprocessing utilities for Organizational-SACU-Mammography.

This package provides reusable components for:

- DICOM loading
- Orientation correction
- Breast-region cropping
- Intensity normalization
- Image quality checks
"""

from .dicom_loading import DICOMImage, load_dicom_image, load_dicom_metadata
from .orientation_correction import correct_mammography_orientation
from .breast_region_cropping import crop_breast_region
from .intensity_normalization import normalize_intensity
from .image_quality_checks import assess_image_quality

__all__ = [
    "DICOMImage",
    "load_dicom_image",
    "load_dicom_metadata",
    "correct_mammography_orientation",
    "crop_breast_region",
    "normalize_intensity",
    "assess_image_quality",
]
```

```python

