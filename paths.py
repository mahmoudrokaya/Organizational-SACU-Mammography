# src/organizational_sacu_mammography/utils/paths.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ProjectPaths:
    """
    Centralized repository path registry.

    The repository uses a reproducible folder structure that separates:

    - raw datasets
    - processed datasets
    - manifests
    - features
    - models
    - evaluation outputs
    - figures
    - logs
    """

    project_root: Path

    configs_dir: Path
    docs_dir: Path
    src_dir: Path

    data_dir: Path
    raw_data_dir: Path
    interim_data_dir: Path
    processed_data_dir: Path

    manifests_dir: Path
    features_dir: Path

    models_dir: Path
    checkpoints_dir: Path

    outputs_dir: Path
    evaluation_dir: Path
    reports_dir: Path
    figures_dir: Path

    logs_dir: Path

    def as_dict(self) -> dict[str, str]:
        """
        Convert paths to string dictionary.
        """

        return {
            field: str(getattr(self, field))
            for field in self.__dataclass_fields__
        }


def ensure_directory(
    directory: str | Path,
) -> Path:
    """
    Ensure that a directory exists.

    Returns
    -------
    Path
        Resolved directory path.
    """

    path = Path(directory).expanduser().resolve()

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def resolve_project_root(
    start_path: Optional[str | Path] = None,
    marker_files: tuple[str, ...] = (
        "README.md",
        "requirements.txt",
        "environment.yml",
    ),
) -> Path:
    """
    Automatically locate repository root.

    Searches upward until one of the repository markers is found.
    """

    if start_path is None:
        current = Path.cwd().resolve()
    else:
        current = Path(start_path).expanduser().resolve()

    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:

        matches = [
            (candidate / marker).exists()
            for marker in marker_files
        ]

        if any(matches):
            return candidate

    raise FileNotFoundError(
        "Unable to locate repository root."
    )


def build_project_paths(
    project_root: str | Path,
    create_directories: bool = True,
) -> ProjectPaths:
    """
    Construct standardized repository paths.
    """

    root = Path(project_root).expanduser().resolve()

    paths = ProjectPaths(
        project_root=root,

        configs_dir=root / "configs",
        docs_dir=root / "docs",
        src_dir=root / "src",

        data_dir=root / "data",
        raw_data_dir=root / "data" / "raw",
        interim_data_dir=root / "data" / "interim",
        processed_data_dir=root / "data" / "processed",

        manifests_dir=root / "outputs" / "manifests",
        features_dir=root / "outputs" / "features",

        models_dir=root / "outputs" / "models",
        checkpoints_dir=root / "outputs" / "models" / "checkpoints",

        outputs_dir=root / "outputs",
        evaluation_dir=root / "outputs" / "evaluation",
        reports_dir=root / "outputs" / "reports",
        figures_dir=root / "outputs" / "figures",

        logs_dir=root / "outputs" / "logs",
    )

    if create_directories:
        create_project_directories(paths)

    return paths


def create_project_directories(
    paths: ProjectPaths,
) -> None:
    """
    Create all standard repository directories.
    """

    for field_name in paths.__dataclass_fields__:

        path = getattr(paths, field_name)

        if isinstance(path, Path):

            if (
                path.suffix == ""
                or not path.exists()
            ):
                ensure_directory(path)


def validate_project_structure(
    paths: ProjectPaths,
) -> None:
    """
    Validate required repository folders.
    """

    required = [
        paths.project_root,
        paths.configs_dir,
        paths.docs_dir,
        paths.src_dir,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing repository components:\n"
            + "\n".join(missing)
        )


def create_experiment_directory(
    paths: ProjectPaths,
    experiment_name: str,
) -> Path:
    """
    Create experiment-specific output directory.
    """

    if not experiment_name.strip():
        raise ValueError(
            "experiment_name cannot be empty."
        )

    experiment_dir = (
        paths.outputs_dir
        / "experiments"
        / experiment_name
    )

    return ensure_directory(experiment_dir)


def create_run_directory(
    experiment_directory: str | Path,
    run_name: str,
) -> Path:
    """
    Create run-specific folder inside an experiment.
    """

    if not run_name.strip():
        raise ValueError(
            "run_name cannot be empty."
        )

    run_dir = (
        Path(experiment_directory)
        / run_name
    )

    return ensure_directory(run_dir)


def build_output_file(
    directory: str | Path,
    filename: str,
) -> Path:
    """
    Construct output file path.
    """

    if not filename:
        raise ValueError(
            "filename cannot be empty."
        )

    directory = ensure_directory(directory)

    return directory / filename


def summarize_project_paths(
    paths: ProjectPaths,
) -> "pd.DataFrame":
    """
    Produce path audit table.
    """

    import pandas as pd

    rows = []

    for field_name in paths.__dataclass_fields__:

        value = getattr(paths, field_name)

        rows.append(
            {
                "path_name": field_name,
                "path": str(value),
                "exists": Path(value).exists(),
            }
        )

    return pd.DataFrame(rows)


def locate_latest_experiment(
    experiments_root: str | Path,
) -> Optional[Path]:
    """
    Locate most recently modified experiment folder.
    """

    root = Path(experiments_root)

    if not root.exists():
        return None

    directories = [
        d
        for d in root.iterdir()
        if d.is_dir()
    ]

    if not directories:
        return None

    return max(
        directories,
        key=lambda x: x.stat().st_mtime,
    )
