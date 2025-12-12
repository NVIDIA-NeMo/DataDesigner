from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, Self, TypeVar, get_args, get_origin

import duckdb
from huggingface_hub import HfFileSystem

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

    Seeds are read using duckdb. Subclasses define duckdb connection setup details
    and how to get a URI that can be queried with duckdb (i.e. "... FROM <uri> ...").

    The Data Designer engine automatically supplies the appropriate config type, along
    with a SecretResolver (available in case any fields in the config are treated as
    secrets that should be resolved).
    """
    config: ConfigT
    secret_resolver: SecretResolver

    @abstractmethod
    def get_dataset_uri(self) -> str: ...

    @abstractmethod
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection: ...

    def attach(self, config: ConfigT, secret_resolver: SecretResolver):
        """
        Called internally by the engine so that these do not need to be provided
        as part of initialization.
        """
        self.config = config
        self.secret_resolver = secret_resolver

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

    def get_dataset_uri(self) -> str:
        return f"file://{self.config.path}"


HF_ENDPOINT = "https://huggingface.co"
_HF_DATASETS_PREFIX = "hf://datasets/"

class HuggingFaceSeedReader(SeedDatasetReader[HuggingFaceSeedConfig]):
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        token = self.secret_resolver.resolve(self.config.token) if self.config.token else None

        # Use skip_instance_cache to avoid fsspec-level caching
        hffs = HfFileSystem(endpoint=HF_ENDPOINT, token=token, skip_instance_cache=True)

        # Clear all internal caches to avoid stale metadata issues
        # HfFileSystem caches file metadata (size, etc.) which can become stale when files are re-uploaded
        if hasattr(hffs, "dircache"):
            hffs.dircache.clear()

        conn = duckdb.connect()
        conn.register_filesystem(hffs)
        return conn

    def get_dataset_uri(self) -> str:
        identifier = self.config.dataset.removeprefix(_HF_DATASETS_PREFIX)
        repo_id, filename = self._get_repo_id_and_filename(identifier)
        return f"{_HF_DATASETS_PREFIX}{repo_id}/{filename}"

    def _get_repo_id_and_filename(self, identifier: str) -> tuple[str, str]:
        """Extract repo_id and filename from identifier."""
        parts = identifier.split("/", 2)
        if len(parts) < 3:
            raise SeedDatasetReaderError(
                "Could not extract repo id and filename from file_id, "
                "expected 'hf://datasets/{repo-namespace}/{repo-name}/{filename}'"
            )
        repo_ns, repo_name, filename = parts
        return f"{repo_ns}/{repo_name}", filename


_DF_TABLE_NAME = "df"

class DataFrameSeedReader(SeedDatasetReader[DataFrameSeedConfig]):
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect()
        conn.register(_DF_TABLE_NAME, self.config.df)
        return conn

    def get_dataset_uri(self) -> str:
        return _DF_TABLE_NAME


DEFAULT_READERS = [
    HuggingFaceSeedReader(),
    LocalFileSeedReader(),
    DataFrameSeedReader(),
]


class SeedDatasetReaderRegistry:
    def __init__(self, readers: Sequence[SeedDatasetReader] = DEFAULT_READERS):
        self._readers: dict[str, SeedDatasetReader] = {}
        for reader in readers or []:
            self.add_reader(reader)

    def add_reader(self, reader: SeedDatasetReader) -> Self:
        seed_type = reader.get_seed_type()

        # Check for conflicts with existing readers
        if seed_type in self._readers:
            raise SeedDatasetReaderError(f"A reader for seed_type {seed_type!r} already exists")

        self._readers[seed_type] = reader
        return self

    def get_reader(
        self, seed_dataset_config: SeedDatasetConfig, secret_resolver: SecretResolver
    ) -> SeedDatasetReader:
        reader = self._get_reader_for_config(seed_dataset_config)
        reader.attach(seed_dataset_config, secret_resolver)
        return reader

    def _get_reader_for_config(self, seed_dataset_config: SeedDatasetConfig) -> SeedDatasetReader:
        seed_type = seed_dataset_config.seed_type
        try:
            return self._readers[seed_type]
        except KeyError:
            raise SeedDatasetReaderError(
                f"No reader found for seed_type {seed_type!r}"
            )
