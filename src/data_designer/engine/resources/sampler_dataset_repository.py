# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
import logging
from pathlib import Path
from typing import IO

from data_designer.config.utils.constants import (
    MANAGED_ASSETS_DIR,
    PERSONAS_DATA_CATALOG_NAME,
)
from data_designer.engine.resources.catalogs import DataCatalog, NemotronPersonasDataCatalog, Table

logger = logging.getLogger(__name__)


class SamplerDatasetRepository(ABC):
    def __init__(self, managed_assets_dir: Path | str = None):
        self._managed_assets_dir = str(managed_assets_dir or MANAGED_ASSETS_DIR)

    @property
    def managed_assets_path(self) -> Path:
        return Path(self._managed_assets_dir)

    @abstractmethod
    def get_data_catalog(self, name: str) -> DataCatalog: ...

    @contextmanager
    def table_reader(self, table_source: str) -> Iterator[IO]:
        with open(table_source, "rb") as fd:
            yield fd

    def get_all_tables(self) -> list[Table]:
        catalogs = list(self._data_catalogs.values())
        return [table for catalog in catalogs for table in catalog.iter_tables()]

    def get_table(self, table_name: str) -> Table:
        if not self.has_access_to_table(table_name):
            raise ValueError(f"Table '{table_name}' does not exist.")
        return next(table for table in self.get_all_tables() if table.name == table_name)

    def has_access_to_data_catalog(self, name: str) -> bool:
        return name in self._data_catalogs and self._data_catalogs[name].num_tables > 0

    def has_access_to_table(self, table_name: str) -> bool:
        return any(table.name == table_name for table in self.get_all_tables())


class LocalSamplerDatasetRepository(SamplerDatasetRepository):
    def __init__(self, managed_assets_dir: Path | str = None):
        super().__init__(managed_assets_dir)
        self._data_catalogs = self._create_default_data_catalogs()

    def get_data_catalog(self, name: str) -> DataCatalog:
        return self._data_catalogs[name]

    def _create_default_data_catalogs(self) -> dict[str, DataCatalog]:
        return {
            PERSONAS_DATA_CATALOG_NAME: NemotronPersonasDataCatalog.create(self.managed_assets_path),
        }
