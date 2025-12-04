# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from huggingface_hub import HfApi, HfFileSystem
import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel, Field

from data_designer.config.errors import InvalidConfigError, InvalidFileFormatError, InvalidFilePathError
from data_designer.config.utils.io_helpers import VALID_DATASET_FILE_EXTENSIONS, validate_path_contains_files_of_type

if TYPE_CHECKING:
    from data_designer.config.seed import SeedDatasetReference

logger = logging.getLogger(__name__)


class DatastoreSettings(BaseModel):
    """Configuration for interacting with a datastore."""

    endpoint: str = Field(
        ...,
        description="Datastore endpoint. Use 'https://huggingface.co' for the Hugging Face Hub.",
    )
    token: Optional[str] = Field(default=None, description="If needed, token to use for authentication.")


def get_file_column_names(file_reference: Union[str, Path, HfFileSystem], file_type: str) -> list[str]:
    """Get column names from a dataset file.

    Args:
        file_reference: Path to the dataset file, or an HfFileSystem object.
        file_type: Type of the dataset file. Must be one of: 'parquet', 'json', 'jsonl', 'csv'.

    Raises:
        InvalidFilePathError: If the file type is not supported.

    Returns:
        List of column names.
    """
    if file_type == "parquet":
        try:
            schema = pq.read_schema(file_reference)
            if hasattr(schema, "names"):
                return schema.names
            else:
                return [field.name for field in schema]
        except Exception as e:
            logger.warning(f"Failed to process parquet file {file_reference}: {e}")
            return []
    elif file_type in ["json", "jsonl"]:
        return pd.read_json(file_reference, orient="records", lines=True, nrows=1).columns.tolist()
    elif file_type == "csv":
        try:
            df = pd.read_csv(file_reference, nrows=1)
            return df.columns.tolist()
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logger.warning(f"Failed to process CSV file {file_reference}: {e}")
            return []
    else:
        raise InvalidFilePathError(f"🛑 Unsupported file type: {file_type!r}")


def fetch_seed_dataset_column_names(seed_dataset_reference: SeedDatasetReference) -> list[str]:
    if hasattr(seed_dataset_reference, "datastore_settings"):
        return fetch_seed_dataset_column_names_from_datastore(
            seed_dataset_reference.repo_id,
            seed_dataset_reference.filename,
            seed_dataset_reference.datastore_settings,
        )
    return fetch_seed_dataset_column_names_from_local_file(seed_dataset_reference.dataset)


def fetch_seed_dataset_column_names_from_datastore(
    repo_id: str,
    filename: str,
    datastore_settings: Optional[Union[DatastoreSettings, dict]] = None,
) -> list[str]:
    file_type = filename.split(".")[-1]
    if f".{file_type}" not in VALID_DATASET_FILE_EXTENSIONS:
        raise InvalidFileFormatError(f"🛑 Unsupported file type: {filename!r}")

    datastore_settings = resolve_datastore_settings(datastore_settings)
    fs = HfFileSystem(endpoint=datastore_settings.endpoint, token=datastore_settings.token, skip_instance_cache=True)

    file_path = _extract_single_file_path_from_glob_pattern_if_present(f"datasets/{repo_id}/{filename}", fs=fs)

    with fs.open(file_path) as f:
        return get_file_column_names(f, file_type)


def fetch_seed_dataset_column_names_from_local_file(dataset_path: str | Path) -> list[str]:
    dataset_path = _validate_dataset_path(dataset_path, allow_glob_pattern=True)
    dataset_path = _extract_single_file_path_from_glob_pattern_if_present(dataset_path)
    return get_file_column_names(dataset_path, str(dataset_path).split(".")[-1])


def resolve_datastore_settings(datastore_settings: DatastoreSettings | dict | None) -> DatastoreSettings:
    if datastore_settings is None:
        raise InvalidConfigError("🛑 Datastore settings are required in order to upload datasets to the datastore.")
    if isinstance(datastore_settings, DatastoreSettings):
        return datastore_settings
    elif isinstance(datastore_settings, dict):
        return DatastoreSettings.model_validate(datastore_settings)
    else:
        raise InvalidConfigError(
            "🛑 Invalid datastore settings format. Must be DatastoreSettings object or dictionary."
        )


def upload_to_hf_hub(
    dataset_path: Union[str, Path],
    filename: str,
    repo_id: str,
    datastore_settings: DatastoreSettings,
    **kwargs,
) -> str:
    datastore_settings = resolve_datastore_settings(datastore_settings)
    dataset_path = _validate_dataset_path(dataset_path)
    filename_ext = filename.split(".")[-1].lower()
    if dataset_path.suffix.lower()[1:] != filename_ext:
        raise InvalidFileFormatError(
            f"🛑 Dataset file extension {dataset_path.suffix!r} does not match `filename` extension .{filename_ext!r}"
        )

    hfapi = HfApi(endpoint=datastore_settings.endpoint, token=datastore_settings.token)
    hfapi.create_repo(repo_id, exist_ok=True, repo_type="dataset")
    hfapi.upload_file(
        path_or_fileobj=dataset_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
        **kwargs,
    )
    return f"{repo_id}/{filename}"


def _extract_single_file_path_from_glob_pattern_if_present(
    file_path: str | Path,
    fs: HfFileSystem | None = None,
) -> Path:
    file_path = Path(file_path)

    # no glob pattern
    if "*" not in str(file_path):
        return file_path

    # glob pattern with HfFileSystem
    if fs is not None:
        file_to_check = None
        file_extension = file_path.name.split(".")[-1]
        for file in fs.ls(str(file_path.parent)):
            filename = file["name"]
            if filename.endswith(f".{file_extension}"):
                file_to_check = filename
        if file_to_check is None:
            raise InvalidFilePathError(f"🛑 No files found matching pattern: {str(file_path)!r}")
        logger.debug(f"Using the first matching file in {str(file_path)!r} to determine column names in seed dataset")
        return Path(file_to_check)

    # glob pattern with local file system
    if not (matching_files := sorted(file_path.parent.glob(file_path.name))):
        raise InvalidFilePathError(f"🛑 No files found matching pattern: {str(file_path)!r}")
    logger.debug(f"Using the first matching file in {str(file_path)!r} to determine column names in seed dataset")
    return matching_files[0]


def _validate_dataset_path(dataset_path: Union[str, Path], allow_glob_pattern: bool = False) -> Path:
    if allow_glob_pattern and "*" in str(dataset_path):
        parts = str(dataset_path).split("*.")
        file_path = parts[0]
        file_extension = parts[-1]
        validate_path_contains_files_of_type(file_path, file_extension)
        return Path(dataset_path)
    if not Path(dataset_path).is_file():
        raise InvalidFilePathError("🛑 To upload a dataset to the datastore, you must provide a valid file path.")
    if not Path(dataset_path).name.endswith(tuple(VALID_DATASET_FILE_EXTENSIONS)):
        raise InvalidFileFormatError(
            "🛑 Dataset files must be in `parquet`, `csv`, or `json` (orient='records', lines=True) format."
        )
    return Path(dataset_path)
