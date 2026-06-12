# src/organizational_sacu_mammography/representation/__init__.py

"""
Representation utilities for Organizational-SACU-Mammography.

This package builds structured mammography representations used by the
Organizational SACU framework, including:

- Multi-view pairing
- Bilateral alignment
- Temporal sequence construction
- Pathway availability masking
- Modeling manifest generation
"""

from .multiview_pairing import (
    MultiViewPairingConfig,
    build_multiview_pairs,
    summarize_multiview_pairs,
)

from .bilateral_alignment import (
    BilateralAlignmentConfig,
    build_bilateral_alignment,
    summarize_bilateral_alignment,
)

from .temporal_sequence_builder import (
    TemporalSequenceConfig,
    build_temporal_sequences,
    summarize_temporal_sequences,
)

from .pathway_masking import (
    PathwayMaskConfig,
    build_pathway_masks,
    summarize_pathway_masks,
)

from .manifest_builder import (
    ManifestSplitConfig,
    build_modeling_manifests,
    summarize_modeling_manifest,
)

__all__ = [
    "MultiViewPairingConfig",
    "build_multiview_pairs",
    "summarize_multiview_pairs",
    "BilateralAlignmentConfig",
    "build_bilateral_alignment",
    "summarize_bilateral_alignment",
    "TemporalSequenceConfig",
    "build_temporal_sequences",
    "summarize_temporal_sequences",
    "PathwayMaskConfig",
    "build_pathway_masks",
    "summarize_pathway_masks",
    "ManifestSplitConfig",
    "build_modeling_manifests",
    "summarize_modeling_manifest",
]
