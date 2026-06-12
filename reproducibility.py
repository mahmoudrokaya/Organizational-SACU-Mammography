# src/organizational_sacu_mammography/utils/reproducibility.py
from __future__ import annotations

import os
import random
import platform
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ReproducibilityConfig:
    """
    Repository-wide reproducibility configuration.

    Used to ensure deterministic execution whenever possible.
    """

    random_seed: int = 42

    deterministic_python_hash_seed: bool = True
    deterministic_numpy: bool = True

    set_python_random_seed: bool = True
    set_numpy_random_seed: bool = True

    enforce_single_thread: bool = False

    omp_num_threads: int = 1
    mkl_num_threads: int = 1
    openblas_num_threads: int = 1

    experiment_name: Optional[str] = None
    experiment_version: Optional[str] = None


def set_global_seed(
    seed: int,
) -> None:
    """
    Set global random seeds.

    This function is intentionally lightweight and dependency-safe.
    """

    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_reproducibility(
    config: ReproducibilityConfig = ReproducibilityConfig(),
) -> None:
    """
    Configure reproducible execution environment.
    """

    if config.deterministic_python_hash_seed:
        os.environ["PYTHONHASHSEED"] = str(
            config.random_seed
        )

    if config.set_python_random_seed:
        random.seed(
            config.random_seed
        )

    if config.set_numpy_random_seed:
        np.random.seed(
            config.random_seed
        )

    if config.enforce_single_thread:

        os.environ["OMP_NUM_THREADS"] = str(
            config.omp_num_threads
        )

        os.environ["MKL_NUM_THREADS"] = str(
            config.mkl_num_threads
        )

        os.environ["OPENBLAS_NUM_THREADS"] = str(
            config.openblas_num_threads
        )


def get_random_state(
    seed: int = 42,
) -> np.random.Generator:
    """
    Create reproducible NumPy random generator.
    """

    return np.random.default_rng(seed)


def capture_environment_metadata() -> Dict[str, Any]:
    """
    Capture execution environment metadata.

    Useful for:
    - experiment tracking
    - reviewer reproducibility requests
    - GitHub releases
    """

    metadata = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }

    return metadata


def build_reproducibility_manifest(
    config: ReproducibilityConfig = ReproducibilityConfig(),
) -> Dict[str, Any]:
    """
    Create complete reproducibility manifest.
    """

    manifest = {
        "reproducibility_configuration": asdict(
            config
        ),
        "environment": capture_environment_metadata(),
    }

    return manifest


def reproducibility_manifest_dataframe(
    config: ReproducibilityConfig = ReproducibilityConfig(),
) -> pd.DataFrame:
    """
    Convert reproducibility information to audit table.
    """

    manifest = build_reproducibility_manifest(
        config
    )

    rows = []

    for section, values in manifest.items():

        if not isinstance(values, dict):
            continue

        for key, value in values.items():

            rows.append(
                {
                    "section": section,
                    "parameter": key,
                    "value": str(value),
                }
            )

    return pd.DataFrame(rows)


def compare_reproducibility_manifests(
    manifest_a: Dict[str, Any],
    manifest_b: Dict[str, Any],
) -> pd.DataFrame:
    """
    Compare two reproducibility manifests.
    """

    rows = []

    sections = sorted(
        set(manifest_a.keys())
        | set(manifest_b.keys())
    )

    for section in sections:

        values_a = manifest_a.get(
            section,
            {},
        )

        values_b = manifest_b.get(
            section,
            {},
        )

        if not isinstance(values_a, dict):
            continue

        if not isinstance(values_b, dict):
            continue

        parameters = sorted(
            set(values_a.keys())
            | set(values_b.keys())
        )

        for parameter in parameters:

            value_a = values_a.get(
                parameter,
                None,
            )

            value_b = values_b.get(
                parameter,
                None,
            )

            rows.append(
                {
                    "section": section,
                    "parameter": parameter,
                    "value_a": str(value_a),
                    "value_b": str(value_b),
                    "match": value_a == value_b,
                }
            )

    return pd.DataFrame(rows)


def validate_reproducibility_configuration(
    config: ReproducibilityConfig,
) -> None:
    """
    Validate reproducibility configuration.
    """

    if config.random_seed < 0:
        raise ValueError(
            "random_seed must be non-negative."
        )

    if config.omp_num_threads <= 0:
        raise ValueError(
            "omp_num_threads must be positive."
        )

    if config.mkl_num_threads <= 0:
        raise ValueError(
            "mkl_num_threads must be positive."
        )

    if config.openblas_num_threads <= 0:
        raise ValueError(
            "openblas_num_threads must be positive."
        )


def describe_reproducibility(
    config: ReproducibilityConfig = ReproducibilityConfig(),
) -> pd.DataFrame:
    """
    Generate compact reproducibility summary.
    """

    rows = [
        {
            "metric": "random_seed",
            "value": config.random_seed,
        },
        {
            "metric": "deterministic_python_hash_seed",
            "value": config.deterministic_python_hash_seed,
        },
        {
            "metric": "deterministic_numpy",
            "value": config.deterministic_numpy,
        },
        {
            "metric": "single_thread_execution",
            "value": config.enforce_single_thread,
        },
        {
            "metric": "experiment_name",
            "value": config.experiment_name,
        },
        {
            "metric": "experiment_version",
            "value": config.experiment_version,
        },
    ]

    return pd.DataFrame(rows)


def export_reproducibility_report(
    output_path: str,
    config: ReproducibilityConfig = ReproducibilityConfig(),
) -> str:
    """
    Export reproducibility report as CSV.
    """

    report = reproducibility_manifest_dataframe(
        config
    )

    report.to_csv(
        output_path,
        index=False,
    )

    return output_path


def reproducible_shuffle(
    dataframe: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Deterministic dataframe shuffle.
    """

    return dataframe.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)


def reproducible_train_test_split_indices(
    n_samples: int,
    test_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate deterministic train/test indices.
    """

    if not 0 < test_fraction < 1:
        raise ValueError(
            "test_fraction must be between 0 and 1."
        )

    rng = np.random.default_rng(seed)

    indices = np.arange(n_samples)

    rng.shuffle(indices)

    split_idx = int(
        n_samples * (1.0 - test_fraction)
    )

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return train_idx, test_idx
