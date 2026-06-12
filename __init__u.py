# src/organizational_sacu_mammography/utils/__init__.py
"""
Utility helpers for Organizational-SACU-Mammography.

This package provides shared repository utilities for:

- Project path resolution
- Safe tabular and JSON I/O
- Logging configuration
- Reproducibility control
"""

from .paths import (
    ProjectPaths,
    resolve_project_root,
    ensure_directory,
    build_project_paths,
)

from .io import (
    read_table,
    write_table,
    read_json,
    write_json,
    write_text,
    read_text,
)

from .logging_utils import (
    configure_logger,
    get_logger,
)

from .reproducibility import (
    set_global_seed,
    ReproducibilityConfig,
    describe_reproducibility,
)

__all__ = [
    "ProjectPaths",
    "resolve_project_root",
    "ensure_directory",
    "build_project_paths",
    "read_table",
    "write_table",
    "read_json",
    "write_json",
    "write_text",
    "read_text",
    "configure_logger",
    "get_logger",
    "set_global_seed",
    "ReproducibilityConfig",
    "describe_reproducibility",
]
