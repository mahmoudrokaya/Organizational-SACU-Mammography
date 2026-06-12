# src/organizational_sacu_mammography/features/__init__.py

"""
Feature extraction utilities for Organizational-SACU-Mammography.

This package provides pathway-specific feature extraction modules for:

- Local-regional tissue descriptors
- Multi-view CC/MLO descriptors
- Bilateral asymmetry descriptors
- Temporal-spatial progression descriptors
- Metadata descriptors
- Full organizational feature-matrix construction
"""

from .local_regional_features import (
    extract_local_regional_features,
    summarize_local_regional_features,
)

from .multiview_features import (
    extract_multiview_features,
    summarize_multiview_features,
)

from .bilateral_features import (
    extract_bilateral_features,
    summarize_bilateral_features,
)

from .temporal_spatial_features import (
    extract_temporal_spatial_features,
    summarize_temporal_spatial_features,
)

from .metadata_features import (
    extract_metadata_features,
    summarize_metadata_features,
)

from .feature_matrix_builder import (
    build_feature_matrix,
    summarize_feature_matrix,
)

__all__ = [
    "extract_local_regional_features",
    "summarize_local_regional_features",
    "extract_multiview_features",
    "summarize_multiview_features",
    "extract_bilateral_features",
    "summarize_bilateral_features",
    "extract_temporal_spatial_features",
    "summarize_temporal_spatial_features",
    "extract_metadata_features",
    "summarize_metadata_features",
    "build_feature_matrix",
    "summarize_feature_matrix",
]
