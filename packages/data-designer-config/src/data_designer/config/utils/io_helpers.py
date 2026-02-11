# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from decimal import Decimal
from numbers import Number
from pathlib import Path, PurePosixPath
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
        dataframe = _maybe_rewrite_url(dataframe)
        # Parse extension from the URL path to avoid query-string contamination (e.g. "csv?token=…").
        ext = PurePosixPath(urlparse(dataframe).path).suffix.lstrip(".").lower()
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
    return _smart_load_yaml_internal(yaml_in, from_url=False)


def _smart_load_yaml_internal(yaml_in: str | Path | dict, *, from_url: bool) -> dict:
    """Internal YAML loader with context to prevent URL recursion on fetched payloads."""
    if isinstance(yaml_in, dict):
        yaml_out = yaml_in
    elif not from_url and isinstance(yaml_in, str) and is_http_url(yaml_in):
        yaml_out = _load_config_from_url(yaml_in)
    elif isinstance(yaml_in, Path) or (isinstance(yaml_in, str) and os.path.isfile(yaml_in)):
        with open(yaml_in) as file:
            yaml_out = yaml.safe_load(file)
    elif isinstance(yaml_in, str):
        if not from_url and yaml_in.endswith((".yaml", ".yml")) and not os.path.isfile(yaml_in):
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


def _maybe_rewrite_url(url: str) -> str:
    """Rewrite known hosting-provider file-view URLs to raw-content URLs."""
    rewritten = _maybe_rewrite_github_url(url)
    if rewritten != url:
        return rewritten
    return _maybe_rewrite_huggingface_hub_url(url)


def _safe_url_for_log(url: str) -> str:
    """Return URL without query/fragment for safe logging."""
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    return f"{parsed.scheme}://{hostname}{parsed.path}"


def _maybe_rewrite_github_url(url: str) -> str:
    """Rewrite GitHub blob URLs to raw.githubusercontent.com equivalents.

    GitHub blob URLs (e.g. https://github.com/org/repo/blob/main/config.yaml)
    serve HTML pages, not raw file content. This rewrites them so that
    downstream fetchers get the actual file.
    """
    parsed = urlparse(url)
    if parsed.hostname not in {"github.com", "www.github.com"}:
        return url

    # GitHub blob path: /{owner}/{repo}/blob/{ref}/{path...}
    # Split on "/" -> ["", owner, repo, "blob", ref, ...path segments]
    segments = parsed.path.split("/")
    if len(segments) >= 5 and segments[3] == "blob":
        raw_path = "/".join(segments[:3] + segments[4:])
        # Preserve query params (e.g. token for private repos), drop fragment
        query = f"?{parsed.query}" if parsed.query else ""
        raw_url = f"https://raw.githubusercontent.com{raw_path}{query}"
        # Strip query params from log output to avoid leaking tokens.
        safe_original = _safe_url_for_log(url)
        safe_rewritten = f"https://raw.githubusercontent.com{raw_path}"
        logger.info("Rewrote GitHub blob URL to raw URL: %s -> %s", safe_original, safe_rewritten)
        return raw_url

    return url


def _maybe_rewrite_huggingface_hub_url(url: str) -> str:
    """Rewrite Hugging Face Hub blob URLs to raw URL equivalents."""
    parsed = urlparse(url)
    if parsed.hostname not in {"huggingface.co", "www.huggingface.co"}:
        return url

    # Hugging Face Hub blob path patterns:
    # - /{namespace}/{repo}/blob/{ref}/{path...}
    # - /datasets/{namespace}/{repo}/blob/{ref}/{path...}
    # - /spaces/{namespace}/{repo}/blob/{ref}/{path...}
    segments = parsed.path.split("/")
    blob_segment_index = 4 if len(segments) >= 5 and segments[1] in {"datasets", "spaces"} else 3
    if len(segments) < blob_segment_index + 3 or segments[blob_segment_index] != "blob":
        return url

    raw_segments = [*segments]
    raw_segments[blob_segment_index] = "raw"
    raw_path = "/".join(raw_segments)
    query = f"?{parsed.query}" if parsed.query else ""
    raw_url = f"https://huggingface.co{raw_path}{query}"

    # Strip query params from log output to avoid leaking tokens.
    safe_original = _safe_url_for_log(url)
    safe_rewritten = f"https://huggingface.co{raw_path}"
    logger.info("Rewrote Hugging Face blob URL to raw URL: %s -> %s", safe_original, safe_rewritten)
    return raw_url


def _raise_for_failed_http_status(url: str, response: requests.Response) -> None:
    """Raise a ValueError with actionable details for failing HTTP status codes."""
    status_code = response.status_code
    if not isinstance(status_code, int) or status_code < 400:
        return

    if status_code == 401:
        raise ValueError(
            f"Failed to fetch config URL '{url}': received 401 Unauthorized. "
            "This URL requires authentication, but authenticated config URL loading is not currently supported."
        )
    if status_code == 403:
        raise ValueError(
            f"Failed to fetch config URL '{url}': received 403 Forbidden. "
            "Ensure you have permission to access this resource."
        )
    if status_code == 404:
        raise ValueError(
            f"Failed to fetch config URL '{url}': received 404 Not Found. Check that the URL path is correct."
        )

    reason = response.reason if isinstance(response.reason, str) and response.reason else "Unknown Error"
    raise ValueError(f"Failed to fetch config URL '{url}': received HTTP {status_code} ({reason}).")


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
    url = _maybe_rewrite_url(url)

    parsed_url = urlparse(url)
    suffix = Path(parsed_url.path).suffix.lower()
    if suffix not in VALID_CONFIG_FILE_EXTENSIONS:
        supported = ", ".join(sorted(VALID_CONFIG_FILE_EXTENSIONS))
        raise ValueError(f"Unsupported config URL extension '{suffix}'. Supported extensions: {supported}")

    try:
        response = requests.get(url, timeout=10)
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch config URL '{url}': {e}") from e

    _raise_for_failed_http_status(url=url, response=response)

    if len(response.content) > MAX_CONFIG_URL_SIZE_BYTES:
        raise ValueError(f"Config from URL '{url}' exceeds maximum size of {MAX_CONFIG_URL_SIZE_BYTES} bytes")

    try:
        content = response.content.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode config from URL '{url}' as UTF-8: {e}") from e

    try:
        return _smart_load_yaml_internal(content, from_url=True)
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
