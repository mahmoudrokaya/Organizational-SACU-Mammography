```python
# src/organizational_sacu_mammography/preprocessing/breast_region_cropping.py

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


def _foreground_mask(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32)

    if x.max() > x.min():
        x_norm = ((x - x.min()) / (x.max() - x.min()) * 255).astype(np.uint8)
    else:
        return np.zeros_like(x, dtype=np.uint8)

    _, mask = cv2.threshold(
        x_norm,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def crop_breast_region(
    image: np.ndarray,
    padding: int = 10,
    min_area_ratio: float = 0.10,
) -> Tuple[np.ndarray, Dict]:
    """
    Crop the breast foreground region using intensity-based segmentation.

    Returns:
        cropped_image, crop_info
    """
    mask = _foreground_mask(image)

    h, w = image.shape[:2]
    area_ratio = float(np.count_nonzero(mask)) / float(h * w)

    if area_ratio < min_area_ratio:
        return image.copy(), {
            "cropped": False,
            "reason": "foreground_area_below_threshold",
            "area_ratio": area_ratio,
            "bbox": [0, 0, h, w],
        }

    ys, xs = np.where(mask > 0)

    y1 = max(int(ys.min()) - padding, 0)
    y2 = min(int(ys.max()) + padding + 1, h)
    x1 = max(int(xs.min()) - padding, 0)
    x2 = min(int(xs.max()) + padding + 1, w)

    cropped = image[y1:y2, x1:x2].copy()

    return cropped, {
        "cropped": True,
        "reason": "ok",
        "area_ratio": area_ratio,
        "bbox": [y1, x1, y2, x2],
        "original_shape": [h, w],
        "cropped_shape": list(cropped.shape),
    }
```

