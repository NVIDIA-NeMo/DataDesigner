# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import duckdb
from huggingface_hub import HfApi, HfFileSystem

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


class LocalSeedDatasetDataStore(SeedDatasetDataStore):
    """Local filesystem-based dataset storage."""

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect()

    def get_dataset_uri(self, file_id: str) -> str:
        return file_id


class HfHubSeedDatasetDataStore(SeedDatasetDataStore):
    """Hugging Face and Data Store dataset storage."""

    def __init__(self, endpoint: str, token: str | None):
        self.hfapi = HfApi(endpoint=endpoint, token=token)
        self.endpoint = endpoint
        self.token = token

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        """Create a DuckDB connection with a fresh HfFileSystem registered.

        Creates a new HfFileSystem instance for each connection to ensure file metadata
        is fetched fresh from the datastore, avoiding cache-related issues when reading
        recently updated parquet files.

        Returns:
            A DuckDB connection with the HfFileSystem registered for hf:// URI support.
        """
        # Use skip_instance_cache to avoid fsspec-level caching
        hffs = HfFileSystem(endpoint=self.endpoint, token=self.token, skip_instance_cache=True)

        # Clear all internal caches to avoid stale metadata issues
        # HfFileSystem caches file metadata (size, etc.) which can become stale when files are re-uploaded
        if hasattr(hffs, "dircache"):
            hffs.dircache.clear()

        conn = duckdb.connect()
        conn.register_filesystem(hffs)
        return conn

    def get_dataset_uri(self, file_id: str) -> str:
        identifier = file_id.removeprefix(_HF_DATASETS_PREFIX)
        repo_id, filename = self._get_repo_id_and_filename(identifier)
        return f"{_HF_DATASETS_PREFIX}{repo_id}/{filename}"

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
