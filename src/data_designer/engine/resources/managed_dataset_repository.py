# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
import logging
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

import duckdb
import pandas as pd

from data_designer.config.utils.constants import LOCALES_WITH_MANAGED_DATASETS
from data_designer.engine.resources.managed_storage import LocalBlobStorageProvider, ManagedBlobStorage

logger = logging.getLogger(__name__)

DATASETS_ROOT = "datasets"
"""
Path in object storage to managed datasets
"""


@dataclass
class Table:
    """
    Managed datasets are organized by dataset by table under a root
    table path in object storage.
    """

    source: str
    """
    Table source path
    """

    schema: str = "main"
    """
    Specifies the schema to use when registering the table.

    Note: this is not the schema of the table, but rather the _database_
    schema to associated with the table.
    """

    @cached_property
    def name(self) -> str:
        return Path(self.source).stem


DataCatalog = list[Table]


# For now we hardcode the remote data catalog in code. This make it easier
# initialize the data catalog. Eventually we can make this work more
# dynamically once this data catalog pattern becomes more widely adopted.
DEFAULT_DATA_CATALOG: DataCatalog = [Table(f"{locale}.parquet") for locale in LOCALES_WITH_MANAGED_DATASETS]


class ManagedDatasetRepository(ABC):
    @abstractmethod
    def query(self, sql: str, parameters: list[Any]) -> pd.DataFrame: ...

    @property
    @abstractmethod
    def data_catalog(self) -> DataCatalog: ...


class DuckDBDatasetRepository(ManagedDatasetRepository):
    """
    Provides a duckdb based sql interface over Gretel managed datasets.
    """

    _default_config = {"threads": 2, "memory_limit": "4 gb"}

    def __init__(
        self,
        blob_storage: ManagedBlobStorage,
        config: dict | None = None,
        data_catalog: DataCatalog = DEFAULT_DATA_CATALOG,
        datasets_root: str = DATASETS_ROOT,
        use_cache: bool = True,
    ):
        """
        Create a new DuckDB backed dataset repository

        Args:
            blob_storage: A managed blob storage provider
            config: DuckDB configuration options,
            https://duckdb.org/docs/configuration/overview.html#configuration-reference
            data_catalog: A list of tables to register with the DuckDB instance
            datasets_root: The root path in blob storage to managed datasets
            use_cache: Whether to cache datasets locally. Trades off disk memory
            and startup time for faster queries.
        """
        self._data_catalog = data_catalog
        self._data_sets_root = datasets_root
        self._blob_storage = blob_storage
        self._config = self._default_config if config is None else config
        self._use_cache = use_cache

        # Configure database and register tables
        self.db = duckdb.connect(config=self._config)

        # Dataset registration completion is tracked with an event. Consumers can
        # wait on this event to ensure the catalog is ready.
        self._registration_event = threading.Event()
        self._register_lock = threading.Lock()

        # Kick off dataset registration in a background thread so that IO-heavy
        # caching and view creation can run asynchronously without blocking the
        # caller that constructs this repository instance.
        self._register_thread = threading.Thread(target=self._register_datasets, daemon=True)
        self._register_thread.start()

    def _register_datasets(self):
        # Just in case this method gets called from inside a thread.
        # This operation isn't thread-safe by default, so we
        # synchronize the registration process.
        if self._registration_event.is_set():
            return
        with self._register_lock:
            # check once more to see if the catalog is ready it's possible a
            # previous thread already registered the dataset.
            if self._registration_event.is_set():
                return
            try:
                for table in self.data_catalog:
                    key = table.source if table.schema == "main" else f"{table.schema}/{table.source}"
                    if self._use_cache:
                        tmp_root = Path(tempfile.gettempdir()) / "dd_cache"
                        local_path = tmp_root / key
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        if not local_path.exists():
                            start = time.time()
                            logger.debug("Caching database %s to %s", table.name, local_path)
                            with self._blob_storage.get_blob(f"{self._data_sets_root}/{key}") as src_fd:
                                with open(local_path, "wb") as dst_fd:
                                    dst_fd.write(src_fd.read())
                            logger.debug(
                                "Cached database %s in %.2f s",
                                table.name,
                                time.time() - start,
                            )
                        data_path = local_path.as_posix()
                    else:
                        data_path = self._blob_storage.uri_for_key(f"{self._data_sets_root}/{key}")
                    if table.schema != "main":
                        self.db.sql(f"CREATE SCHEMA IF NOT EXISTS {table.schema}")
                    logger.debug(f"Registering dataset {table.name} from {data_path}")
                    self.db.sql(f"CREATE VIEW {table.schema}.{table.name} AS FROM '{data_path}'")

                logger.debug("DuckDBDatasetRepository registration complete")

            except Exception as e:
                logger.exception(f"Failed to register datasets: {str(e)}")

            finally:
                # Signal that registration is complete so any waiting queries can proceed.
                self._registration_event.set()

    def query(self, sql: str, parameters: list[Any]) -> pd.DataFrame:
        # Ensure dataset registration has completed. Possible future optimization:
        # pull datasets in parallel and only wait here if the query requires a
        # table that isn't cached.
        if not self._registration_event.is_set():
            logger.debug("Waiting for dataset caching and registration to finish...")
            self._registration_event.wait()

        # the duckdb connection isn't thread-safe, so we create a new
        # connection per query using cursor().
        # more details here: https://duckdb.org/docs/stable/guides/python/multiple_threads.html
        cursor = self.db.cursor()
        try:
            df = cursor.execute(sql, parameters).df()
        finally:
            cursor.close()
        return df

    @property
    def data_catalog(self) -> DataCatalog:
        return self._data_catalog


def load_managed_dataset_repository(blob_storage: ManagedBlobStorage, locales: list[str]) -> ManagedDatasetRepository:
    return DuckDBDatasetRepository(
        blob_storage,
        config={"threads": 1, "memory_limit": "2 gb"},
        data_catalog=[Table(f"{locale}.parquet") for locale in locales],
        # Only cache if not using local storage.
        use_cache=not isinstance(blob_storage, LocalBlobStorageProvider),
    )
