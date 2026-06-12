# src/organizational_sacu_mammography/utils/io.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .paths import ensure_directory


SUPPORTED_TABLE_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".parquet",
    ".xlsx",
}


def _resolve_path(
    path: str | Path,
) -> Path:
    return Path(path).expanduser().resolve()


def _validate_input_file(
    path: str | Path,
) -> Path:
    file_path = _resolve_path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Expected file but received directory: {file_path}"
        )

    return file_path


def _prepare_output_file(
    path: str | Path,
) -> Path:
    file_path = _resolve_path(path)

    ensure_directory(file_path.parent)

    return file_path


def read_table(
    path: str | Path,
    **kwargs,
) -> pd.DataFrame:
    """
    Read tabular data.

    Supported formats:
    - CSV
    - TSV
    - Parquet
    - XLSX
    """

    file_path = _validate_input_file(path)

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(
            file_path,
            **kwargs,
        )

    if suffix == ".tsv":
        return pd.read_csv(
            file_path,
            sep="\t",
            **kwargs,
        )

    if suffix == ".parquet":
        return pd.read_parquet(
            file_path,
            **kwargs,
        )

    if suffix == ".xlsx":
        return pd.read_excel(
            file_path,
            **kwargs,
        )

    raise ValueError(
        f"Unsupported table format: {suffix}"
    )


def write_table(
    dataframe: pd.DataFrame,
    path: str | Path,
    index: bool = False,
    **kwargs,
) -> Path:
    """
    Write tabular data.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    file_path = _prepare_output_file(path)

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        dataframe.to_csv(
            file_path,
            index=index,
            **kwargs,
        )

    elif suffix == ".tsv":
        dataframe.to_csv(
            file_path,
            sep="\t",
            index=index,
            **kwargs,
        )

    elif suffix == ".parquet":
        dataframe.to_parquet(
            file_path,
            index=index,
            **kwargs,
        )

    elif suffix == ".xlsx":
        dataframe.to_excel(
            file_path,
            index=index,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unsupported table format: {suffix}"
        )

    return file_path


class NumpyJSONEncoder(json.JSONEncoder):
    """
    JSON encoder supporting NumPy and pandas types.
    """

    def default(
        self,
        obj: Any,
    ) -> Any:

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()

        if pd.isna(obj):
            return None

        return super().default(obj)


def read_json(
    path: str | Path,
) -> Dict[str, Any]:
    """
    Read JSON file.
    """

    file_path = _validate_input_file(path)

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def write_json(
    data: Dict[str, Any],
    path: str | Path,
    indent: int = 4,
    sort_keys: bool = True,
) -> Path:
    """
    Write JSON file.
    """

    file_path = _prepare_output_file(path)

    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            cls=NumpyJSONEncoder,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=False,
        )

    return file_path


def read_text(
    path: str | Path,
    encoding: str = "utf-8",
) -> str:
    """
    Read text file.
    """

    file_path = _validate_input_file(path)

    with open(
        file_path,
        "r",
        encoding=encoding,
    ) as f:
        return f.read()


def write_text(
    text: str,
    path: str | Path,
    encoding: str = "utf-8",
) -> Path:
    """
    Write text file.
    """

    file_path = _prepare_output_file(path)

    with open(
        file_path,
        "w",
        encoding=encoding,
    ) as f:
        f.write(text)

    return file_path


def append_text(
    text: str,
    path: str | Path,
    encoding: str = "utf-8",
) -> Path:
    """
    Append text to existing file.
    """

    file_path = _prepare_output_file(path)

    with open(
        file_path,
        "a",
        encoding=encoding,
    ) as f:
        f.write(text)

    return file_path


def read_manifest(
    path: str | Path,
) -> pd.DataFrame:
    """
    Read SACU manifest file.
    """

    manifest = read_table(path)

    if manifest.empty:
        raise ValueError(
            "Manifest file is empty."
        )

    return manifest


def write_manifest(
    manifest_df: pd.DataFrame,
    path: str | Path,
) -> Path:
    """
    Write standardized SACU manifest.
    """

    required_columns = {
        "patient_id",
    }

    missing = required_columns - set(
        manifest_df.columns
    )

    if missing:
        raise ValueError(
            f"Manifest missing required columns: {sorted(missing)}"
        )

    return write_table(
        manifest_df,
        path,
        index=False,
    )


def dataframe_memory_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate dataframe memory audit.
    """

    rows = []

    total_bytes = 0

    for column in dataframe.columns:

        memory_bytes = int(
            dataframe[column].memory_usage(
                deep=True
            )
        )

        total_bytes += memory_bytes

        rows.append(
            {
                "column": column,
                "dtype": str(
                    dataframe[column].dtype
                ),
                "memory_bytes": memory_bytes,
            }
        )

    rows.append(
        {
            "column": "__TOTAL__",
            "dtype": "",
            "memory_bytes": total_bytes,
        }
    )

    return pd.DataFrame(rows)


def dataset_audit_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce repository-wide dataset audit summary.
    """

    rows = [
        {
            "metric": "n_rows",
            "value": int(len(dataframe)),
        },
        {
            "metric": "n_columns",
            "value": int(
                dataframe.shape[1]
            ),
        },
        {
            "metric": "missing_values",
            "value": int(
                dataframe.isna().sum().sum()
            ),
        },
        {
            "metric": "duplicate_rows",
            "value": int(
                dataframe.duplicated().sum()
            ),
        },
        {
            "metric": "memory_mb",
            "value": float(
                dataframe.memory_usage(
                    deep=True
                ).sum()
                / (1024**2)
            ),
        },
    ]

    return pd.DataFrame(rows)


def save_experiment_artifacts(
    artifacts: Dict[str, Any],
    output_directory: str | Path,
) -> Dict[str, Path]:
    """
    Save experiment outputs automatically.

    DataFrames -> CSV
    Dictionaries -> JSON
    Strings -> TXT
    """

    output_dir = ensure_directory(
        output_directory
    )

    written: Dict[str, Path] = {}

    for name, value in artifacts.items():

        if isinstance(value, pd.DataFrame):

            path = output_dir / f"{name}.csv"

            write_table(
                value,
                path,
                index=False,
            )

            written[name] = path

        elif isinstance(value, dict):

            path = output_dir / f"{name}.json"

            write_json(
                value,
                path,
            )

            written[name] = path

        elif isinstance(value, str):

            path = output_dir / f"{name}.txt"

            write_text(
                value,
                path,
            )

            written[name] = path

    return written
