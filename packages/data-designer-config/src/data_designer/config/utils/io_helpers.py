# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from decimal import Decimal
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import requests
import yaml

from data_designer.config.errors import InvalidFileFormatError, InvalidFilePathError
from data_designer.lazy_heavy_imports import np, pd

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)

MAX_CONFIG_URL_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB
VALID_DATASET_FILE_EXTENSIONS = {".parquet", ".csv", ".json", ".jsonl"}
VALID_CONFIG_FILE_EXTENSIONS = {".yaml", ".yml", ".json"}


def ensure_config_dir_exists(config_dir: Path) -> None:
    """Create configuration directory if it doesn't exist.

    Args:
        config_dir: Directory path to create
    """
    config_dir.mkdir(parents=True, exist_ok=True)


def load_config_file(file_path: Path) -> dict:
    """Load a YAML configuration file.

    Args:
        file_path: Path to the YAML file

    Returns:
        Parsed YAML content as dictionary

    Raises:
        InvalidFilePathError: If file doesn't exist
        InvalidFileFormatError: If YAML is malformed
        InvalidConfigError: If file is empty
    """
    from data_designer.config.errors import InvalidConfigError

    if not file_path.exists():
        raise InvalidFilePathError(f"Configuration file not found: {file_path}")

    try:
        with open(file_path) as f:
            content = yaml.safe_load(f)

        if content is None:
            raise InvalidConfigError(f"Configuration file is empty: {file_path}")

        return content

    except yaml.YAMLError as e:
        raise InvalidFileFormatError(f"Invalid YAML format in {file_path}: {e}")


def save_config_file(file_path: Path, config: dict) -> None:
    """Save configuration to a YAML file.

    Args:
        file_path: Path where to save the file
        config: Configuration dictionary to save

    Raises:
        IOError: If file cannot be written
    """
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        yaml.safe_dump(
            config,
            f,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
            allow_unicode=True,
        )


def read_parquet_dataset(path: Path) -> pd.DataFrame:
    """Read a parquet dataset from a path.

    Args:
        path: The path to the parquet dataset, can be either a file or a directory.

    Returns:
        The parquet dataset as a pandas DataFrame.
    """
    try:
        return pd.read_parquet(path, dtype_backend="pyarrow")
    except Exception as e:
        if path.is_dir() and "Unsupported cast" in str(e):
            logger.warning("Failed to read parquets as folder, falling back to individual files")
            return pd.concat(
                [pd.read_parquet(file, dtype_backend="pyarrow") for file in sorted(path.glob("*.parquet"))],
                ignore_index=True,
            )
        else:
            raise e


def validate_dataset_file_path(file_path: str | Path, should_exist: bool = True) -> Path:
    """Validate that a dataset file path has a valid extension and optionally exists.

    Args:
        file_path: The path to validate, either as a string or Path object.
        should_exist: If True, verify that the file exists. Defaults to True.
    Returns:
        The validated path as a Path object.
    Raises:
        InvalidFilePathError: If the path is not a file.
        InvalidFileFormatError: If the path does not have a valid extension.
    """
    file_path = Path(file_path)
    if should_exist and not Path(file_path).is_file():
        raise InvalidFilePathError(f"🛑 Path {file_path} is not a file.")
    if not file_path.name.lower().endswith(tuple(VALID_DATASET_FILE_EXTENSIONS)):
        raise InvalidFileFormatError(
            "🛑 Dataset files must be in parquet, csv, or jsonl/json (orient='records', lines=True) format."
        )
    return file_path


def validate_path_contains_files_of_type(path: str | Path, file_extension: str) -> None:
    """Validate that a path contains files of a specific type.

    Args:
        path: The path to validate. Can contain wildcards like `*.parquet`.
        file_extension: The extension of the files to validate (without the dot, e.g., "parquet").
    Returns:
        None if the path contains files of the specified type, raises an error otherwise.
    Raises:
        InvalidFilePathError: If the path does not contain files of the specified type.
    """
    if not any(Path(path).glob(f"*.{file_extension}")):
        raise InvalidFilePathError(f"🛑 Path {path!r} does not contain files of type {file_extension!r}.")


def smart_load_dataframe(dataframe: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load a dataframe from file if a path is given, otherwise return the dataframe.

    Args:
        dataframe: A path to a file or a pandas DataFrame object.

    Returns:
        A pandas DataFrame object.
    """
    if isinstance(dataframe, pd.DataFrame):
        return dataframe

    # Get the file extension.
    if isinstance(dataframe, str) and dataframe.startswith("http"):
        ext = dataframe.split(".")[-1].lower()
    else:
        dataframe = Path(dataframe)
        ext = dataframe.suffix.lower()
        if not dataframe.exists():
            raise FileNotFoundError(f"File not found: {dataframe}")

    # Load the dataframe based on the file extension.
    if ext == "csv":
        return pd.read_csv(dataframe)
    elif ext == "json":
        return pd.read_json(dataframe, lines=True)
    elif ext == "parquet":
        return pd.read_parquet(dataframe)
    else:
        raise ValueError(f"Unsupported file format: {dataframe}")


def smart_load_yaml(yaml_in: str | Path | dict) -> dict:
    """Return the yaml config as a dict given flexible input types.

    Args:
        config: The config as a dict, yaml string, or yaml file path.

    Returns:
        The config as a dict.
    """
    if isinstance(yaml_in, dict):
        yaml_out = yaml_in
    elif isinstance(yaml_in, str) and is_http_url(yaml_in):
        yaml_out = _load_config_from_url(yaml_in)
    elif isinstance(yaml_in, Path) or (isinstance(yaml_in, str) and os.path.isfile(yaml_in)):
        with open(yaml_in) as file:
            yaml_out = yaml.safe_load(file)
    elif isinstance(yaml_in, str):
        if yaml_in.endswith((".yaml", ".yml")) and not os.path.isfile(yaml_in):
            raise FileNotFoundError(f"File not found: {yaml_in}")
        else:
            yaml_out = yaml.safe_load(yaml_in)
    else:
        raise ValueError(
            f"'{yaml_in}' is an invalid yaml config format. Valid options are: dict, yaml string, or yaml file path."
        )

    if not isinstance(yaml_out, dict):
        raise ValueError(f"Loaded yaml must be a dict, got {type(yaml_out).__name__}.")

    return yaml_out


def is_http_url(value: str) -> bool:
    """Check whether a string is an HTTP or HTTPS URL."""
    parsed_url = urlparse(value)
    return parsed_url.scheme in {"http", "https"} and bool(parsed_url.netloc)


def _load_config_from_url(url: str) -> dict:
    """Fetch a remote YAML/JSON config URL and return the parsed dict.

    Args:
        url: HTTP(S) URL pointing to a YAML or JSON configuration file.

    Returns:
        The parsed configuration as a dictionary.

    Raises:
        ValueError: If the URL extension is unsupported, the fetch fails,
            the response exceeds the size limit, or parsing produces a
            non-dict result.
    """
    parsed_url = urlparse(url)
    suffix = Path(parsed_url.path).suffix.lower()
    if suffix not in VALID_CONFIG_FILE_EXTENSIONS:
        supported = ", ".join(sorted(VALID_CONFIG_FILE_EXTENSIONS))
        raise ValueError(f"Unsupported config URL extension '{suffix}'. Supported extensions: {supported}")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch config URL '{url}': {e}") from e

    if len(response.content) > MAX_CONFIG_URL_SIZE_BYTES:
        raise ValueError(f"Config from URL '{url}' exceeds maximum size of {MAX_CONFIG_URL_SIZE_BYTES} bytes")

    try:
        content = response.content.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode config from URL '{url}' as UTF-8: {e}") from e

    try:
        return smart_load_yaml(content)
    except (yaml.YAMLError, ValueError) as e:
        raise ValueError(f"Failed to parse config from URL '{url}': {e}") from e


def serialize_data(data: dict | list | str | Number, **kwargs) -> str:
    if isinstance(data, dict):
        return json.dumps(data, ensure_ascii=False, default=_convert_to_serializable, **kwargs)
    elif isinstance(data, list):
        return json.dumps(data, ensure_ascii=False, default=_convert_to_serializable, **kwargs)
    elif isinstance(data, str):
        return data
    elif isinstance(data, Number):
        return str(data)
    else:
        raise ValueError(f"Invalid data type: {type(data)}")


def _convert_to_serializable(obj: Any) -> Any:
    """Convert non-JSON-serializable objects to JSON-serializable Python-native types.

    Raises:
        TypeError: If the object type is not supported for serialization.
    """
    if isinstance(obj, (set, list)):
        return list(obj)
    if isinstance(obj, (pd.Series, np.ndarray)):
        return obj.tolist()

    if pd.isna(obj):
        return None

    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, timedelta):
        return obj.total_seconds()
    if isinstance(obj, (np.datetime64, np.timedelta64)):
        return str(obj)

    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()

    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")

    # Unsupported type
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
