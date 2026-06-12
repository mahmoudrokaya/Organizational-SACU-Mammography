```python
# src/organizational_sacu_mammography/metadata/__init__.py

"""
Metadata utilities for Organizational-SACU-Mammography.

This package provides reusable components for:

- Metadata harmonization
- Patient identifier mapping
- Diagnostic label mapping
- VinDr-Mammo audit utilities
"""

from .metadata_harmonization import (
    MetadataHarmonizer,
    harmonize_metadata,
)

from .patient_id_mapping import (
    PatientIDMapper,
    create_patient_id_map,
)

from .label_mapping import (
    LabelMapper,
    map_binary_labels,
)

from .vindr_audit import (
    VinDrAuditReport,
    audit_vindr_metadata,
)

__all__ = [
    "MetadataHarmonizer",
    "harmonize_metadata",
    "PatientIDMapper",
    "create_patient_id_map",
    "LabelMapper",
    "map_binary_labels",
    "VinDrAuditReport",
    "audit_vindr_metadata",
]
```


