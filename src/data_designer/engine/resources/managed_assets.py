# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import IO

from typing_extensions import Self

from data_designer.config.utils.constants import (
    LOCALES_WITH_MANAGED_DATASETS,
    MANAGED_ASSETS_PATH,
    PERSONAS_DATA_CATALOG_NAME,
)

logger = logging.getLogger(__name__)


@dataclass
class Table:
    source: str
    name: str


class DataCatalog:
    """A data catalog is a collection of tables."""

    def __init__(self, name: str, tables: list[Table]):
        self.name = name
        self.tables = tables

    def __iter__(self) -> Iterator[Table]:
        return iter(self.tables)

    def __len__(self) -> int:
        return len(self.tables)


class NemotronPersonasDataCatalog(DataCatalog):
    @classmethod
    def create(cls, managed_assets_path: str | Path) -> Self:
        tables = []
        locale_version_map = _create_locale_version_map(Path(managed_assets_path))
        for locale, version in locale_version_map.items():
            tables.append(
                Table(
                    source=f"{managed_assets_path}/{_get_dataset_name(locale, version)}/*.parquet",
                    name=f"{locale.lower()}_{version}",
                )
            )
        return cls(name=PERSONAS_DATA_CATALOG_NAME, tables=tables)


class DatasetManager(ABC):
    def __init__(self, managed_assets_dir: Path | str = None):
        self._managed_assets_dir = str(managed_assets_dir or MANAGED_ASSETS_PATH)

    @property
    def managed_assets_path(self) -> Path:
        return Path(self._managed_assets_dir)

    @abstractmethod
    def get_data_catalog(self, name: str) -> DataCatalog: ...

    @contextmanager
    def table_reader(self, table_source: str) -> Iterator[IO]:
        with open(table_source, "rb") as fd:
            yield fd

    def get_data_catalogs(self, names: list[str], flatten: bool = False) -> list[DataCatalog]:
        if isinstance(names, str):
            raise ValueError("`names` must be a list of strings")

        catalogs = [self.get_data_catalog(name) for name in names]
        if flatten:
            return [table for catalog in catalogs for table in catalog]
        return catalogs

    def get_table(self, catalog_name: str, table_name: str, exact_match: bool = False) -> Table:
        if not self.has_access_to_table(table_name, exact_match):
            raise ValueError(f"Table {table_name} not found in the dataset manager.")

        catalog = self.get_data_catalog(catalog_name)
        return next(table for table in catalog if _match_table_name(table, table_name, exact_match))

    def has_access_to_data_catalog(self, name: str) -> bool:
        return name in self._data_catalogs and len(self._data_catalogs[name]) > 0

    def has_access_to_table(self, table_name: str, exact_match: bool = False) -> bool:
        return any(
            _match_table_name(table, table_name, exact_match)
            for catalog in self._data_catalogs.values()
            for table in catalog
        )


class LocalDatasetManager(DatasetManager):
    def __init__(self, managed_assets_dir: Path | str = None):
        super().__init__(managed_assets_dir)
        self._data_catalogs = self._create_default_data_catalogs()

    def get_data_catalog(self, name: str) -> DataCatalog:
        return self._data_catalogs[name]

    def _create_default_data_catalogs(self) -> dict[str, DataCatalog]:
        return {
            PERSONAS_DATA_CATALOG_NAME: NemotronPersonasDataCatalog.create(self.managed_assets_path),
        }


def _create_locale_version_map(managed_assets_path: str | Path) -> dict[str, str]:
    locales = set()
    for locale in LOCALES_WITH_MANAGED_DATASETS:
        if list(managed_assets_path.glob(f"{_get_dataset_name(locale)}*")):
            locales.add(locale.lower())

    locale_version_map = {}
    for locale in locales:
        locale_version_map[locale] = sorted(managed_assets_path.glob(f"{PERSONAS_DATA_CATALOG_NAME}-{locale}*"))[
            -1
        ].name.split("-")[-1]
    return locale_version_map


def _get_dataset_name(locale: str, version: str | None = None) -> str:
    return f"{PERSONAS_DATA_CATALOG_NAME}-{locale.lower()}" + (f"-{version}" if version else "")


def _match_table_name(table: Table, table_name: str, exact_match: bool) -> bool:
    return table.name == table_name if exact_match else table.name.startswith(table_name)
