# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Generic, TypeVar

from huggingface_hub import HfFileSystem
from typing_extensions import Self

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.seed_source import (
    HuggingFaceSeedSource,
    LocalFileSeedSource,
    SeedSource,
)
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.registry.errors import NotFoundInRegistryError
from data_designer.engine.registry.handler import Handler, HandlerRegistry
from data_designer.engine.secret_resolver import SecretResolver
from data_designer.errors import DataDesignerError
from data_designer.plugins.plugin import PluginType
from data_designer.plugins.registry import PluginRegistry

if TYPE_CHECKING:
    import duckdb


class SeedReaderError(DataDesignerError): ...


SourceT = TypeVar("ConfigT", bound=SeedSource)


class SeedReader(Handler[SourceT], ABC, Generic[SourceT]):
    """Base class for reading a seed dataset.

    Seeds are read using duckdb. Reader implementations define duckdb connection setup details
    and how to get a URI that can be queried with duckdb (i.e. "... FROM <uri> ...").

    The Data Designer engine automatically supplies the appropriate SeedSource
    and a SecretResolver to use for any secret fields in the config.
    """

    source: SourceT
    secret_resolver: SecretResolver

    @abstractmethod
    def get_dataset_uri(self) -> str: ...

    @abstractmethod
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection: ...

    def attach(self, source: SourceT, secret_resolver: SecretResolver):
        """Attach a source and secret resolver to the instance.

        This is called internally by the engine so that these objects do not
        need to be provided in the reader's constructor.
        """
        self.source = source
        self.secret_resolver = secret_resolver

    def get_column_names(self) -> list[str]:
        """Returns the seed dataset's column names"""
        conn = self.create_duckdb_connection()
        describe_query = f"DESCRIBE SELECT * FROM '{self.get_dataset_uri()}'"
        column_descriptions = conn.execute(describe_query).fetchall()
        return [col[0] for col in column_descriptions]

    def get_seed_type(self) -> str:
        """Return the seed_type of the source class this reader is generic over."""
        return type(self).get_registered_name("seed_type")


class LocalFileSeedReader(SeedReader[LocalFileSeedSource]):
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        return lazy.duckdb.connect()

    def get_dataset_uri(self) -> str:
        return self.source.path


class HuggingFaceSeedReader(SeedReader[HuggingFaceSeedSource]):
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        token = self.secret_resolver.resolve(self.source.token) if self.source.token else None

        # Use skip_instance_cache to avoid fsspec-level caching
        hffs = HfFileSystem(endpoint=self.source.endpoint, token=token, skip_instance_cache=True)

        # Clear all internal caches to avoid stale metadata issues
        # HfFileSystem caches file metadata (size, etc.) which can become stale when files are re-uploaded
        if hasattr(hffs, "dircache"):
            hffs.dircache.clear()

        conn = lazy.duckdb.connect()
        conn.register_filesystem(hffs)
        return conn

    def get_dataset_uri(self) -> str:
        return f"hf://{self.source.path}"


class DataFrameSeedReader(SeedReader[DataFrameSeedSource]):
    # This is a "magic string" that gets registered in the duckdb connection to make the dataframe directly queryable.
    _table_name = "df"

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        conn = lazy.duckdb.connect()
        conn.register(self._table_name, self.source.df)
        return conn

    def get_dataset_uri(self) -> str:
        return self._table_name


class SeedReaderRegistry(HandlerRegistry[SeedSource, SeedReader[SeedSource]]):
    def __init__(self, readers: Sequence[type[SeedReader[SeedSource]] | SeedReader[SeedSource]]):
        super().__init__(
            discriminator_field="seed_type",
            handler_factory=lambda reader_type, _: reader_type(),
            error_type=SeedReaderError,
            handler_label="reader",
        )
        self._reader_instances: dict[str, SeedReader[SeedSource]] = {}
        for reader in readers:
            self.add_reader(reader)

    def add_reader(self, reader: type[SeedReader[SeedSource]] | SeedReader[SeedSource]) -> Self:
        if isinstance(reader, SeedReader):
            try:
                seed_type = type(reader).get_registered_name("seed_type")
            except (TypeError, ValueError) as error:
                raise SeedReaderError(str(error)) from error
            if self.has_registered_name(seed_type):
                raise SeedReaderError(f"A reader for seed_type {seed_type!r} already exists")
            self.register(type(reader))
            self._reader_instances[seed_type] = reader
            return self

        self.register(reader)
        return self

    def get_reader(self, seed_dataset_source: SeedSource, secret_resolver: SecretResolver) -> SeedReader:
        seed_type = self.get_name_for_config(seed_dataset_source)
        reader = self._reader_instances.get(seed_type)
        if reader is None:
            try:
                reader = self.create_for_config(seed_dataset_source)
            except NotFoundInRegistryError as error:
                raise SeedReaderError(f"No reader found for seed_type {seed_type!r}") from error
        reader.attach(seed_dataset_source, secret_resolver)
        return reader


def create_default_seed_reader_registry(with_plugins: bool = True) -> SeedReaderRegistry:
    readers: list[type[SeedReader[SeedSource]]] = [
        HuggingFaceSeedReader,
        LocalFileSeedReader,
        DataFrameSeedReader,
    ]
    if with_plugins:
        for plugin in PluginRegistry().get_plugins(PluginType.SEED_READER):
            readers.append(plugin.impl_cls)
    return SeedReaderRegistry(readers=readers)
