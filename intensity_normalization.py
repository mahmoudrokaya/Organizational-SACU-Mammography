```python
# src/organizational_sacu_mammography/preprocessing/intensity_normalization.py

from __future__ import annotations

import numpy as np


def percentile_clip(
    image: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.0,
) -> np.ndarray:
    image = image.astype(np.float32, copy=False)

    low = np.percentile(image, lower)
    high = np.percentile(image, upper)

    if high <= low:
        return image.copy()

    return np.clip(image, low, high)


def normalize_intensity(
    image: np.ndarray,
    method: str = "minmax",
    clip_percentiles: bool = True,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize mammography image intensity.

    Supported methods:
    - minmax: scale to [0, 1]
    - zscore: zero mean and unit variance
    """
    x = image.astype(np.float32, copy=True)

    if clip_percentiles:
        x = percentile_clip(x, lower_percentile, upper_percentile)

    method = method.lower()

    if method == "minmax":
        xmin = float(np.min(x))
        xmax = float(np.max(x))
        if xmax - xmin < eps:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - xmin) / (xmax - xmin + eps)).astype(np.float32)

    if method == "zscore":
        mean = float(np.mean(x))
        std = float(np.std(x))
        if std < eps:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - mean) / (std + eps)).astype(np.float32)

    raise ValueError(f"Unsupported normalization method: {method}")
```
