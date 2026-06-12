# src/organizational_sacu_mammography/utils/logging_utils.py
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .paths import ensure_directory


DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | "
    "%(name)s | %(filename)s:%(lineno)d | %(message)s"
)

DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass
class LoggingConfig:
    """
    Repository-wide logging configuration.
    """

    logger_name: str = "organizational_sacu_mammography"

    level: str = "INFO"

    console_logging: bool = True
    file_logging: bool = True

    log_filename: str = "organizational_sacu_mammography.log"

    log_format: str = DEFAULT_LOG_FORMAT
    date_format: str = DEFAULT_DATE_FORMAT

    overwrite_existing_handlers: bool = True


def _resolve_log_level(
    level: str,
) -> int:
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    level_upper = level.upper()

    if level_upper not in level_map:
        raise ValueError(
            f"Unsupported log level: {level}"
        )

    return level_map[level_upper]


def _build_formatter(
    config: LoggingConfig,
) -> logging.Formatter:
    return logging.Formatter(
        fmt=config.log_format,
        datefmt=config.date_format,
    )


def _clear_handlers(
    logger: logging.Logger,
) -> None:
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def configure_logger(
    log_directory: str | Path,
    config: LoggingConfig = LoggingConfig(),
) -> logging.Logger:
    """
    Configure repository logger.

    Creates:
    - Console logger
    - File logger

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(
        config.logger_name
    )

    logger.setLevel(
        _resolve_log_level(config.level)
    )

    logger.propagate = False

    if (
        config.overwrite_existing_handlers
        and logger.handlers
    ):
        _clear_handlers(logger)

    formatter = _build_formatter(
        config
    )

    if config.console_logging:

        console_handler = logging.StreamHandler(
            sys.stdout
        )

        console_handler.setFormatter(
            formatter
        )

        console_handler.setLevel(
            _resolve_log_level(config.level)
        )

        logger.addHandler(
            console_handler
        )

    if config.file_logging:

        log_dir = ensure_directory(
            log_directory
        )

        log_file = (
            Path(log_dir)
            / config.log_filename
        )

        file_handler = logging.FileHandler(
            log_file,
            encoding="utf-8",
        )

        file_handler.setFormatter(
            formatter
        )

        file_handler.setLevel(
            _resolve_log_level(config.level)
        )

        logger.addHandler(
            file_handler
        )

    return logger


def get_logger(
    name: str,
) -> logging.Logger:
    """
    Retrieve child logger.
    """

    return logging.getLogger(name)


def create_experiment_logger(
    experiment_name: str,
    log_directory: str | Path,
    level: str = "INFO",
) -> logging.Logger:
    """
    Create experiment-specific logger.
    """

    config = LoggingConfig(
        logger_name=experiment_name,
        level=level,
        log_filename=f"{experiment_name}.log",
    )

    return configure_logger(
        log_directory=log_directory,
        config=config,
    )


def log_dataframe_summary(
    logger: logging.Logger,
    dataframe,
    dataframe_name: str,
) -> None:
    """
    Log dataframe summary statistics.
    """

    try:
        logger.info(
            "DataFrame '%s' shape=%s",
            dataframe_name,
            dataframe.shape,
        )

        logger.info(
            "DataFrame '%s' columns=%d",
            dataframe_name,
            dataframe.shape[1],
        )

        logger.info(
            "DataFrame '%s' missing_values=%d",
            dataframe_name,
            int(
                dataframe.isna().sum().sum()
            ),
        )

    except Exception as exc:
        logger.warning(
            "Unable to summarize dataframe '%s': %s",
            dataframe_name,
            exc,
        )


def log_feature_matrix_summary(
    logger: logging.Logger,
    X,
    matrix_name: str,
) -> None:
    """
    Log modeling matrix summary.
    """

    try:
        logger.info(
            "%s shape=%s",
            matrix_name,
            X.shape,
        )

        logger.info(
            "%s features=%d",
            matrix_name,
            X.shape[1],
        )

    except Exception as exc:
        logger.warning(
            "Unable to summarize matrix '%s': %s",
            matrix_name,
            exc,
        )


def log_class_distribution(
    logger: logging.Logger,
    y,
    target_name: str = "target",
) -> None:
    """
    Log class distribution.
    """

    try:
        import pandas as pd

        counts = (
            pd.Series(y)
            .value_counts()
            .sort_index()
        )

        logger.info(
            "Class distribution for '%s'",
            target_name,
        )

        for label, count in counts.items():
            logger.info(
                "Class=%s Count=%d",
                label,
                int(count),
            )

    except Exception as exc:
        logger.warning(
            "Unable to log class distribution: %s",
            exc,
        )


def log_configuration(
    logger: logging.Logger,
    configuration,
    configuration_name: str,
) -> None:
    """
    Log dataclass configuration values.
    """

    try:

        if hasattr(
            configuration,
            "__dataclass_fields__",
        ):
            values = {
                field: getattr(
                    configuration,
                    field,
                )
                for field in configuration.__dataclass_fields__
            }

        elif isinstance(
            configuration,
            dict,
        ):
            values = configuration

        else:
            values = vars(configuration)

        logger.info(
            "Configuration: %s",
            configuration_name,
        )

        for key, value in values.items():
            logger.info(
                "%s=%s",
                key,
                value,
            )

    except Exception as exc:
        logger.warning(
            "Unable to log configuration '%s': %s",
            configuration_name,
            exc,
        )


def log_exception(
    logger: logging.Logger,
    exception: Exception,
    message: Optional[str] = None,
) -> None:
    """
    Log exception with traceback.
    """

    if message is None:
        message = str(exception)

    logger.exception(message)


def summarize_log_file(
    log_file: str | Path,
):
    """
    Build simple log audit table.
    """

    import pandas as pd

    log_path = Path(log_file)

    if not log_path.exists():
        raise FileNotFoundError(
            f"Log file not found: {log_path}"
        )

    levels = {
        "DEBUG": 0,
        "INFO": 0,
        "WARNING": 0,
        "ERROR": 0,
        "CRITICAL": 0,
    }

    with open(
        log_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            for level in levels:
                if f"| {level} |" in line:
                    levels[level] += 1

    return pd.DataFrame(
        [
            {
                "log_level": level,
                "count": count,
            }
            for level, count in levels.items()
        ]
    )
