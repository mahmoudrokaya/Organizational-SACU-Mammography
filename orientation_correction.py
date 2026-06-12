```python
# src/organizational_sacu_mammography/preprocessing/orientation_correction.py

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def correct_photometric_interpretation(
    image: np.ndarray,
    metadata: Dict,
) -> np.ndarray:
    """
    Convert MONOCHROME1 images to MONOCHROME2-like representation.

    In MONOCHROME1, higher values correspond to darker pixels.
    """
    photometric = str(metadata.get("photometric_interpretation", "")).upper()

    corrected = image.astype(np.float32, copy=True)

    if photometric == "MONOCHROME1":
        corrected = corrected.max() - corrected

    return corrected


def correct_mammography_orientation(
    image: np.ndarray,
    metadata: Dict,
    standardize_laterality: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Apply basic mammography orientation correction.

    The function preserves the image content but standardizes right-breast
    images by horizontal flipping when requested. This makes downstream
    regional descriptors more comparable across laterality.
    """
    corrected = correct_photometric_interpretation(image, metadata)

    laterality = (
        metadata.get("image_laterality")
        or metadata.get("laterality")
        or ""
    )
    laterality = str(laterality).upper()

    applied = {
        "photometric_corrected": str(metadata.get("photometric_interpretation", "")).upper() == "MONOCHROME1",
        "horizontal_flip_applied": False,
    }

    if standardize_laterality and laterality in {"R", "RIGHT"}:
        corrected = np.fliplr(corrected)
        applied["horizontal_flip_applied"] = True

    return corrected, applied
```


