# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Generic, TypeVar, get_args, get_origin

from huggingface_hub import HfFileSystem
from typing_extensions import Self

from data_designer.config.seed_source import (
    DataFrameSeedSource,
    HuggingFaceSeedSource,
    LocalFileSeedSource,
    SeedSource,
)
from data_designer.engine.secret_resolver import SecretResolver
from data_designer.errors import DataDesignerError
from data_designer.lazy_heavy_imports import duckdb

if TYPE_CHECKING:
    import duckdb


class SeedReaderError(DataDesignerError): ...


SourceT = TypeVar("ConfigT", bound=SeedSource)


class SeedReader(ABC, Generic[SourceT]):
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
        # Get the generic type arguments from the reader class
        # Check __orig_bases__ for the generic base class
        for base in getattr(type(self), "__orig_bases__", []):
            origin = get_origin(base)
            if origin is SeedReader:
                args = get_args(base)
                if args:
                    source_cls = args[0]
                    # Extract seed_type from the source class
                    if hasattr(source_cls, "model_fields") and "seed_type" in source_cls.model_fields:
                        field = source_cls.model_fields["seed_type"]
                        default_value = field.default
                        if isinstance(default_value, str):
                            return default_value

        raise SeedReaderError("Reader does not have a valid generic source type with seed_type")


class LocalFileSeedReader(SeedReader[LocalFileSeedSource]):
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect()

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

        conn = duckdb.connect()
        conn.register_filesystem(hffs)
        return conn

    def get_dataset_uri(self) -> str:
        return f"hf://{self.source.path}"


class DataFrameSeedReader(SeedReader[DataFrameSeedSource]):
    # This is a "magic string" that gets registered in the duckdb connection to make the dataframe directly queryable.
    _table_name = "df"

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect()
        conn.register(self._table_name, self.source.df)
        return conn

    def get_dataset_uri(self) -> str:
        return self._table_name


class SeedReaderRegistry:
    def __init__(self, readers: Sequence[SeedReader]):
        self._readers: dict[str, SeedReader] = {}
        for reader in readers:
            self.add_reader(reader)

    def add_reader(self, reader: SeedReader) -> Self:
        seed_type = reader.get_seed_type()

        if seed_type in self._readers:
            raise SeedReaderError(f"A reader for seed_type {seed_type!r} already exists")

        self._readers[seed_type] = reader
        return self

    def get_reader(self, seed_dataset_source: SeedSource, secret_resolver: SecretResolver) -> SeedReader:
        reader = self._get_reader_for_source(seed_dataset_source)
        reader.attach(seed_dataset_source, secret_resolver)
        return reader

    def _get_reader_for_source(self, seed_dataset_source: SeedSource) -> SeedReader:
        seed_type = seed_dataset_source.seed_type
        try:
            return self._readers[seed_type]
        except KeyError:
            raise SeedReaderError(f"No reader found for seed_type {seed_type!r}")
