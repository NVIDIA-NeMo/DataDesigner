# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import data_designer.lazy_heavy_imports as lazy

if TYPE_CHECKING:
    import duckdb
    import pandas as pd

logger = logging.getLogger(__name__)

DATASETS_ROOT = "datasets"
THREADS = 1
MEMORY_LIMIT = "2 gb"


class PersonReader(ABC):
    """Provides duckdb access to managed datasets (e.g., person name data).

    Implementations control connection creation (custom fsspec clients, caching, etc.)
    and URI resolution. Modeled after SeedReader.

    The ``locale`` parameter passed to ``get_dataset_uri`` is a logical identifier
    (e.g., ``"en_US"``). Each implementation decides how to map that to a physical
    URI -- the caller does not construct paths or add file extensions.
    """

    @abstractmethod
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection: ...

    @abstractmethod
    def get_dataset_uri(self, locale: str) -> str: ...

    def execute(self, query: str, parameters: list[list]) -> pd.DataFrame:
        conn = self.create_duckdb_connection()
        cursor = conn.cursor()
        try:
            return cursor.execute(query, parameters).df()
        finally:
            cursor.close()
            conn.close()


class LocalPersonReader(PersonReader):
    """Reads person datasets from a local filesystem path."""

    def __init__(self, root_path: Path) -> None:
        self._root_path = root_path

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        return lazy.duckdb.connect(config={"threads": THREADS, "memory_limit": MEMORY_LIMIT})

    def get_dataset_uri(self, locale: str) -> str:
        return f"{self._root_path}/{DATASETS_ROOT}/{locale}.parquet"


def create_person_reader(assets_storage: str) -> PersonReader:
    path = Path(assets_storage)
    if not path.exists():
        raise RuntimeError(f"Local storage path {assets_storage!r} does not exist.")

    logger.debug(f"Using local storage for managed datasets: {assets_storage!r}")
    return LocalPersonReader(path)
