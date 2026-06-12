```python
# src/organizational_sacu_mammography/preprocessing/image_quality_checks.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np


@dataclass
class ImageQualityReport:
    valid: bool
    height: int
    width: int
    min_value: float
    max_value: float
    mean_value: float
    std_value: float
    nonzero_fraction: float
    failure_reason: str | None = None

    def to_dict(self) -> Dict:
        return asdict(self)


def assess_image_quality(
    image: np.ndarray,
    min_height: int = 128,
    min_width: int = 128,
    min_nonzero_fraction: float = 0.01,
    min_std: float = 1e-6,
) -> ImageQualityReport:
    """
    Basic image-quality checks for mammography preprocessing.
    """
    if image is None:
        return ImageQualityReport(
            valid=False,
            height=0,
            width=0,
            min_value=0.0,
            max_value=0.0,
            mean_value=0.0,
            std_value=0.0,
            nonzero_fraction=0.0,
            failure_reason="image_is_none",
        )

    if image.ndim != 2:
        return ImageQualityReport(
            valid=False,
            height=int(image.shape[0]) if image.ndim > 0 else 0,
            width=int(image.shape[1]) if image.ndim > 1 else 0,
            min_value=float(np.min(image)),
            max_value=float(np.max(image)),
            mean_value=float(np.mean(image)),
            std_value=float(np.std(image)),
            nonzero_fraction=float(np.count_nonzero(image) / image.size),
            failure_reason="image_not_2d",
        )

    h, w = image.shape
    nonzero_fraction = float(np.count_nonzero(image) / image.size)
    std_value = float(np.std(image))

    valid = True
    reason = None

    if h < min_height or w < min_width:
        valid = False
        reason = "image_too_small"
    elif nonzero_fraction < min_nonzero_fraction:
        valid = False
        reason = "too_few_nonzero_pixels"
    elif std_value < min_std:
        valid = False
        reason = "near_constant_image"

    return ImageQualityReport(
        valid=valid,
        height=int(h),
        width=int(w),
        min_value=float(np.min(image)),
        max_value=float(np.max(image)),
        mean_value=float(np.mean(image)),
        std_value=std_value,
        nonzero_fraction=nonzero_fraction,
        failure_reason=reason,
    )
```

