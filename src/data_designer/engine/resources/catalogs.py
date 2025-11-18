# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from dataclasses import dataclass
import logging
from pathlib import Path

from typing_extensions import Self

from data_designer.config.utils.constants import (
    LOCALES_WITH_MANAGED_DATASETS,
    PERSONAS_DATA_CATALOG_NAME,
)

logger = logging.getLogger(__name__)


@dataclass
class Table:
    source: str
    name: str
    version: str | None = None


class DataCatalog:
    """A data catalog is a collection of tables."""

    def __init__(self, name: str, tables: list[Table]):
        self.name = name
        self.tables = tables

    @property
    def num_tables(self) -> int:
        return len(self.tables)

    def iter_tables(self) -> Iterator[Table]:
        return iter(self.tables)


class NemotronPersonasDataCatalog(DataCatalog):
    @classmethod
    def create(cls, managed_assets_path: str | Path) -> Self:
        tables = []
        locale_version_map = _create_locale_version_map(Path(managed_assets_path))
        for locale, version in locale_version_map.items():
            tables.append(
                Table(
                    source=f"{managed_assets_path}/{_get_dataset_name(locale, version)}/*.parquet",
                    name=locale.lower(),
                    version=version,
                )
            )
        return cls(name=PERSONAS_DATA_CATALOG_NAME, tables=tables)


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
