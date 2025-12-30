# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar, get_args, get_origin

import duckdb
from huggingface_hub import HfFileSystem
from typing_extensions import Self

from data_designer.config.seed_dataset import (
    DataFrameSeedConfig,
    HuggingFaceSeedConfig,
    LocalFileSeedConfig,
    SeedDatasetConfig,
)
from data_designer.engine.secret_resolver import SecretResolver
from data_designer.errors import DataDesignerError


class SeedDatasetReaderError(DataDesignerError): ...


ConfigT = TypeVar("ConfigT", bound=SeedDatasetConfig)


class SeedDatasetReader(ABC, Generic[ConfigT]):
    """Base class for reading a seed dataset.

    Seeds are read using duckdb. Reader implementations define duckdb connection setup details
    and how to get a URI that can be queried with duckdb (i.e. "... FROM <uri> ...").

    The Data Designer engine automatically supplies the appropriate SeedDatasetConfig
    and a SecretResolver to use for any secret fields in the config.
    """

    config: ConfigT
    secret_resolver: SecretResolver

    @abstractmethod
    def get_dataset_uri(self) -> str: ...

    @abstractmethod
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection: ...

    def attach(self, config: ConfigT, secret_resolver: SecretResolver):
        """Attach a config and secret resolver to the instance.

        This is called internally by the engine so that these objects do not
        need to be provided in the reader's constructor.
        """
        self.config = config
        self.secret_resolver = secret_resolver

    def get_column_names(self) -> list[str]:
        """Returns the seed dataset's column names"""
        conn = self.create_duckdb_connection()
        describe_query = f"DESCRIBE SELECT * FROM '{self.get_dataset_uri()}'"
        column_descriptions = conn.execute(describe_query).fetchall()
        return [col[0] for col in column_descriptions]

    def get_seed_type(self) -> str:
        """Return the seed_type of the config class this reader is generic over."""
        # Get the generic type arguments from the reader class
        # Check __orig_bases__ for the generic base class
        for base in getattr(type(self), "__orig_bases__", []):
            origin = get_origin(base)
            if origin is SeedDatasetReader:
                args = get_args(base)
                if args:
                    config_cls = args[0]
                    # Extract seed_type from the config class
                    if hasattr(config_cls, "model_fields") and "seed_type" in config_cls.model_fields:
                        field = config_cls.model_fields["seed_type"]
                        default_value = field.default
                        if isinstance(default_value, str):
                            return default_value

        raise SeedDatasetReaderError("Reader does not have a valid generic config type with seed_type")


class LocalFileSeedReader(SeedDatasetReader[LocalFileSeedConfig]):
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect()

    # TODO: should this just be `self.config.path`?
    def get_dataset_uri(self) -> str:
        return f"file://{self.config.path}"


_HF_ENDPOINT = "https://huggingface.co"


class HuggingFaceSeedReader(SeedDatasetReader[HuggingFaceSeedConfig]):
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        token = self.secret_resolver.resolve(self.config.token) if self.config.token else None

        # Use skip_instance_cache to avoid fsspec-level caching
        hffs = HfFileSystem(endpoint=_HF_ENDPOINT, token=token, skip_instance_cache=True)

        # Clear all internal caches to avoid stale metadata issues
        # HfFileSystem caches file metadata (size, etc.) which can become stale when files are re-uploaded
        if hasattr(hffs, "dircache"):
            hffs.dircache.clear()

        conn = duckdb.connect()
        conn.register_filesystem(hffs)
        return conn

    def get_dataset_uri(self) -> str:
        return self.config.dataset


_DF_TABLE_NAME = "df"


class DataFrameSeedReader(SeedDatasetReader[DataFrameSeedConfig]):
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect()
        conn.register(_DF_TABLE_NAME, self.config.df)
        return conn

    def get_dataset_uri(self) -> str:
        return _DF_TABLE_NAME


class SeedDatasetReaderRegistry:
    def __init__(self, readers: Sequence[SeedDatasetReader]):
        self._readers: dict[str, SeedDatasetReader] = {}
        for reader in readers:
            self.add_reader(reader)

    def add_reader(self, reader: SeedDatasetReader) -> Self:
        seed_type = reader.get_seed_type()

        if seed_type in self._readers:
            raise SeedDatasetReaderError(f"A reader for seed_type {seed_type!r} already exists")

        self._readers[seed_type] = reader
        return self

    def get_reader(self, seed_dataset_config: SeedDatasetConfig, secret_resolver: SecretResolver) -> SeedDatasetReader:
        reader = self._get_reader_for_config(seed_dataset_config)
        reader.attach(seed_dataset_config, secret_resolver)
        return reader

    def _get_reader_for_config(self, seed_dataset_config: SeedDatasetConfig) -> SeedDatasetReader:
        seed_type = seed_dataset_config.seed_type
        try:
            return self._readers[seed_type]
        except KeyError:
            raise SeedDatasetReaderError(f"No reader found for seed_type {seed_type!r}")
