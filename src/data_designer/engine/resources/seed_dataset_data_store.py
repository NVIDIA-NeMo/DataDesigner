# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import os
import tempfile

from datasets import DatasetDict, load_dataset
import duckdb
from huggingface_hub import HfApi, HfFileSystem
import pandas as pd

from data_designer.config.utils.io_helpers import validate_dataset_file_path
from data_designer.logging import quiet_noisy_logger

quiet_noisy_logger("httpx")

_HF_DATASETS_PREFIX = "hf://datasets/"


class MalformedFileIdError(Exception):
    """Raised when file_id format is invalid."""


class SeedDatasetDataStore(ABC):
    """Abstract base class for dataset storage implementations."""

    @abstractmethod
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection: ...

    @abstractmethod
    def get_dataset_uri(self, file_id: str) -> str: ...

    @abstractmethod
    def load_dataset(self, file_id: str) -> pd.DataFrame: ...


class LocalSeedDatasetDataStore(SeedDatasetDataStore):
    """Local filesystem-based dataset storage."""

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect()

    def get_dataset_uri(self, file_id: str) -> str:
        return file_id

    def load_dataset(self, file_id: str) -> pd.DataFrame:
        filepath = validate_dataset_file_path(file_id)
        match filepath.suffix.lower():
            case ".csv":
                return pd.read_csv(filepath)
            case ".parquet":
                return pd.read_parquet(filepath)
            case ".json":
                return pd.read_json(filepath, lines=True)
            case ".jsonl":
                return pd.read_json(filepath, lines=True)
            case _:
                raise ValueError("Local datasets must be CSV, Parquet, JSON, or JSONL")


class HfHubSeedDatasetDataStore(SeedDatasetDataStore):
    """Hugging Face and Data Store dataset storage."""

    def __init__(self, endpoint: str, token: str | None):
        self.hfapi = HfApi(endpoint=endpoint, token=token)
        self.hffs = HfFileSystem(endpoint=endpoint, token=token)

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect()
        conn.register_filesystem(self.hffs)
        return conn

    def get_dataset_uri(self, file_id: str) -> str:
        identifier = file_id.removeprefix(_HF_DATASETS_PREFIX)
        repo_id, filename = self._get_repo_id_and_filename(identifier)
        return f"{_HF_DATASETS_PREFIX}{repo_id}/{filename}"

    def load_dataset(self, file_id: str) -> pd.DataFrame:
        identifier = file_id.removeprefix(_HF_DATASETS_PREFIX)
        repo_id, filename = self._get_repo_id_and_filename(identifier)
        is_file = "." in file_id.split("/")[-1]

        self._validate_repo(repo_id)

        if is_file:
            self._validate_file(repo_id, filename)
            return self._download_and_load_file(repo_id, filename)
        else:
            return self._download_and_load_directory(repo_id, filename)

    def _validate_repo(self, repo_id: str) -> None:
        """Validate that the repository exists and is a dataset repo."""
        if not self.hfapi.repo_exists(repo_id, repo_type="dataset"):
            if self.hfapi.repo_exists(repo_id, repo_type="model"):
                raise FileNotFoundError(f"Repo {repo_id} is a model repo, not a dataset repo")
            raise FileNotFoundError(f"Repo {repo_id} does not exist")

    def _validate_file(self, repo_id: str, filename: str) -> None:
        """Validate that the file exists in the repository."""
        if not self.hfapi.file_exists(repo_id, filename, repo_type="dataset"):
            raise FileNotFoundError(f"File {filename} does not exist in repo {repo_id}")

    def _download_and_load_file(self, repo_id: str, filename: str) -> pd.DataFrame:
        """Download a specific file and load it as a dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.hfapi.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=temp_dir,
                repo_type="dataset",
            )
            return self._load_local_dataset(temp_dir)

    def _download_and_load_directory(self, repo_id: str, directory: str) -> pd.DataFrame:
        """Download entire repo and load from specific subdirectory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.hfapi.snapshot_download(
                repo_id=repo_id,
                local_dir=temp_dir,
                repo_type="dataset",
            )
            dataset_path = os.path.join(temp_dir, directory)
            if not os.path.exists(dataset_path):
                dataset_path = temp_dir
            return self._load_local_dataset(dataset_path)

    def _get_repo_id_and_filename(self, identifier: str) -> tuple[str, str]:
        """Extract repo_id and filename from identifier."""
        parts = identifier.split("/", 2)
        if len(parts) < 3:
            raise MalformedFileIdError(
                "Could not extract repo id and filename from file_id, "
                "expected 'hf://datasets/{repo-namespace}/{repo-name}/{filename}'"
            )
        repo_ns, repo_name, filename = parts
        return f"{repo_ns}/{repo_name}", filename

    def _load_local_dataset(self, path: str) -> pd.DataFrame:
        """Load dataset from local path."""
        hf_dataset = load_dataset(path=path)
        if isinstance(hf_dataset, DatasetDict):
            hf_dataset = hf_dataset[list(hf_dataset.keys())[0]]
        return hf_dataset.to_pandas()
