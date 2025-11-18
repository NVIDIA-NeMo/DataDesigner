# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import logging
from pathlib import Path
import tempfile
import threading
import time

import duckdb
import pandas as pd

from data_designer.config.utils.constants import PERSONAS_DATA_CATALOG_NAME
from data_designer.engine.resources.managed_assets import DataCatalog, DatasetManager, LocalDatasetManager

logger = logging.getLogger(__name__)


class ManagedDatasetRepository(ABC):
    @abstractmethod
    def query(self, sql: str) -> pd.DataFrame: ...

    @property
    @abstractmethod
    def data_catalog(self) -> DataCatalog: ...


class DuckDBDatasetRepository(ManagedDatasetRepository):
    """
    Provides a duckdb based sql interface over Gretel managed datasets.
    """

    _default_config = {"threads": 1, "memory_limit": "2 gb"}

    def __init__(
        self,
        dataset_manager: DatasetManager,
        data_catalog_names: list[str],
        config: dict | None = None,
        use_cache: bool = True,
    ):
        """
        Create a new DuckDB backed dataset repository

        Args:
            dataset_manager: A dataset manager
            data_catalog_names: A list of data catalog names to register with the DuckDB instance
            config: DuckDB configuration options,
            https://duckdb.org/docs/configuration/overview.html#configuration-reference
            use_cache: Whether to cache datasets locally. Trades off disk memory
            and startup time for faster queries.
        """
        self._dataset_manager = dataset_manager
        self._config = self._default_config if config is None else config
        self._use_cache = use_cache
        self._data_catalog_names = data_catalog_names

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
                for table in self._dataset_manager.get_data_catalogs(self._data_catalog_names, flatten=True):
                    if self._use_cache:
                        tmp_root = Path(tempfile.gettempdir()) / "dd_cache"
                        local_path = tmp_root / table.name
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        if not local_path.exists():
                            start = time.time()
                            logger.debug("Caching database %s to %s", table.name, local_path)
                            with self._dataset_manager.table_reader(table.source) as src_fd:
                                with open(local_path, "wb") as dst_fd:
                                    dst_fd.write(src_fd.read())
                            logger.debug(
                                "Cached database %s in %.2f s",
                                table.name,
                                time.time() - start,
                            )
                        data_path = local_path.as_posix()
                    else:
                        data_path = table.source
                    logger.debug(f"Registering dataset {table.name} from {data_path}")
                    self.db.sql(f"CREATE VIEW '{table.name}' AS FROM '{data_path}'")

                logger.debug("DuckDBDatasetRepository registration complete")

            except Exception as e:
                logger.exception(f"Failed to register datasets: {str(e)}")

            finally:
                # Signal that registration is complete so any waiting queries can proceed.
                self._registration_event.set()

    def query(self, sql: str) -> pd.DataFrame:
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
            df = cursor.sql(sql).df()
        finally:
            cursor.close()
        return df

    @property
    def data_catalog(self) -> DataCatalog:
        return self._dataset_manager.get_data_catalog()


def create_dataset_repository(
    dataset_manager: DatasetManager,
    data_catalog_names: list[str] | None = None,
) -> ManagedDatasetRepository:
    return DuckDBDatasetRepository(
        dataset_manager,
        data_catalog_names=data_catalog_names or [PERSONAS_DATA_CATALOG_NAME],
        use_cache=not isinstance(dataset_manager, LocalDatasetManager),
    )
