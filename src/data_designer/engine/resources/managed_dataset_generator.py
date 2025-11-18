# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import logging
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

import duckdb
import pandas as pd

from data_designer.engine.resources.sampler_dataset_repository import (
    LocalSamplerDatasetRepository,
    SamplerDatasetRepository,
)

logger = logging.getLogger(__name__)


class ManagedDatasetGenerator(ABC):
    @abstractmethod
    def query(self, sql: str) -> pd.DataFrame: ...


class DuckDBDatasetGenerator(ManagedDatasetGenerator):
    """
    Provides a duckdb based sql interface over Gretel managed datasets.
    """

    _default_config = {"threads": 1, "memory_limit": "2 gb"}

    def __init__(
        self, dataset_repository: SamplerDatasetRepository, config: dict | None = None, use_cache: bool = True
    ):
        """
        Create a new DuckDB backed dataset repository

        Args:
            dataset_repository: A dataset manager
            config: DuckDB configuration options,
            https://duckdb.org/docs/configuration/overview.html#configuration-reference
            use_cache: Whether to cache datasets locally. Trades off disk memory
            and startup time for faster queries.
        """
        self._dataset_repository = dataset_repository
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
                for table in self._dataset_repository.get_all_tables():
                    if self._use_cache:
                        tmp_root = Path(tempfile.gettempdir()) / "dd_cache"
                        local_path = tmp_root / table.name
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        if not local_path.exists():
                            start = time.time()
                            logger.debug("Caching database %s to %s", table.name, local_path)
                            with self._dataset_repository.table_reader(table.source) as src_fd:
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

    def generate_samples_from_table(
        self,
        table_name: str,
        size: int = 1,
        evidence: dict[str, Any | list[Any]] = {},
    ) -> pd.DataFrame:
        query = f"select * from '{table_name}'"
        parameters = []
        if evidence:
            where_conditions = []
            for column, values in evidence.items():
                if values:
                    values = values if isinstance(values, list) else [values]
                    formatted_values = ["?"] * len(values)
                    condition = f"{column} IN ({', '.join(formatted_values)})"
                    where_conditions.append(condition)
                    parameters.extend(values)
            if where_conditions:
                query += " where " + " and ".join(where_conditions)
        query += f" order by random() limit {size}"
        return self.query(query, parameters)


def create_managed_dataset_generator(dataset_repository: SamplerDatasetRepository) -> ManagedDatasetGenerator:
    return DuckDBDatasetGenerator(
        dataset_repository,
        use_cache=not isinstance(dataset_repository, LocalSamplerDatasetRepository),
    )
