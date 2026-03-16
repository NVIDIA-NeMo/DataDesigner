# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Generic, TypeVar, get_args, get_origin

from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from huggingface_hub import HfFileSystem
from typing_extensions import Self

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.seed_source import (
    DirectorySeedSource,
    FileContentsSeedSource,
    FileSystemSeedSource,
    HuggingFaceSeedSource,
    LocalFileSeedSource,
    SeedSource,
)
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.secret_resolver import SecretResolver
from data_designer.errors import DataDesignerError

if TYPE_CHECKING:
    import duckdb
    import pandas as pd


class SeedReaderError(DataDesignerError): ...


@dataclass(frozen=True)
class SeedReaderFileSystemContext:
    fs: AbstractFileSystem
    root_path: Path


SourceT = TypeVar("SourceT", bound=SeedSource)
FileSystemSourceT = TypeVar("FileSystemSourceT", bound=FileSystemSeedSource)


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

    def attach(self, source: SourceT, secret_resolver: SecretResolver) -> None:
        """Attach a source and secret resolver to the instance.

        This is called internally by the engine so that these objects do not
        need to be provided in the reader's constructor.
        """
        self.source = source
        self.secret_resolver = secret_resolver

    def create_dataframe_duckdb_connection(
        self,
        *,
        table_name: str,
        dataframe: pd.DataFrame,
    ) -> duckdb.DuckDBPyConnection:
        conn = lazy.duckdb.connect()
        conn.register(table_name, dataframe)
        return conn

    def create_filesystem_context(self, root_path: Path | str) -> SeedReaderFileSystemContext:
        """Create a rooted filesystem context for directory-backed seed readers."""
        resolved_root_path = Path(root_path).expanduser().resolve()
        rooted_fs = DirFileSystem(path=str(resolved_root_path), fs=LocalFileSystem())
        return SeedReaderFileSystemContext(fs=rooted_fs, root_path=resolved_root_path)

    def get_matching_relative_paths(
        self,
        *,
        context: SeedReaderFileSystemContext,
        file_pattern: str,
        recursive: bool,
    ) -> list[str]:
        max_depth = None if recursive else 1
        relative_paths = [
            _normalize_relative_path(path) for path in context.fs.find("", withdirs=False, maxdepth=max_depth)
        ]
        matched_paths = [
            relative_path
            for relative_path in relative_paths
            if fnmatchcase(PurePosixPath(relative_path).name, file_pattern)
        ]
        matched_paths.sort()

        if not matched_paths:
            search_scope = "under" if recursive else "directly under"
            raise SeedReaderError(f"No files matched file_pattern {file_pattern!r} {search_scope} {context.root_path}")

        return matched_paths

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
            if isinstance(origin, type) and issubclass(origin, SeedReader):
                args = get_args(base)
                if args:
                    source_cls = get_origin(args[0]) or args[0]
                    # Extract seed_type from the source class
                    if hasattr(source_cls, "model_fields") and "seed_type" in source_cls.model_fields:
                        field = source_cls.model_fields["seed_type"]
                        default_value = field.default
                        if isinstance(default_value, str):
                            return default_value

        raise SeedReaderError("Reader does not have a valid generic source type with seed_type")


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
        return self.create_dataframe_duckdb_connection(table_name=self._table_name, dataframe=self.source.df)

    def get_dataset_uri(self) -> str:
        return self._table_name


class FileSystemSeedReader(SeedReader[FileSystemSourceT], ABC):
    _table_name: str

    def __init__(self) -> None:
        self._normalized_df: pd.DataFrame | None = None

    def attach(self, source: FileSystemSourceT, secret_resolver: SecretResolver) -> None:
        self._normalized_df = None
        super().attach(source, secret_resolver)

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        return self.create_dataframe_duckdb_connection(
            table_name=self._table_name,
            dataframe=self._get_normalized_dataframe(),
        )

    def get_dataset_uri(self) -> str:
        return self._table_name

    @abstractmethod
    def get_normalized_records(self, *, context: SeedReaderFileSystemContext) -> list[dict[str, Any]]: ...

    def _get_normalized_dataframe(self) -> pd.DataFrame:
        if self._normalized_df is not None:
            return self._normalized_df

        context = self.create_filesystem_context(self.source.path)
        normalized_records = self.get_normalized_records(context=context)
        if not normalized_records:
            raise SeedReaderError(f"Seed source at {self.source.path} did not produce any rows")

        self._normalized_df = lazy.pd.DataFrame(normalized_records)
        return self._normalized_df


class DirectorySeedReader(FileSystemSeedReader[DirectorySeedSource]):
    _table_name = "directory_df"

    def get_normalized_records(self, *, context: SeedReaderFileSystemContext) -> list[dict[str, Any]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [
            _build_metadata_record(
                context=context,
                relative_path=relative_path,
                source_kind="directory_file",
            )
            for relative_path in matched_paths
        ]


class FileContentsSeedReader(FileSystemSeedReader[FileContentsSeedSource]):
    _table_name = "file_contents_df"

    def get_normalized_records(self, *, context: SeedReaderFileSystemContext) -> list[dict[str, Any]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        records: list[dict[str, Any]] = []

        for relative_path in matched_paths:
            absolute_path = context.root_path / relative_path
            try:
                with context.fs.open(relative_path, "r", encoding=self.source.encoding) as handle:
                    content = handle.read()
            except UnicodeDecodeError as error:
                raise SeedReaderError(
                    f"Failed to decode file {absolute_path} using encoding {self.source.encoding!r}: {error}"
                ) from error
            except OSError as error:
                raise SeedReaderError(f"Failed to read file {absolute_path}: {error}") from error

            record = _build_metadata_record(
                context=context,
                relative_path=relative_path,
                source_kind="file_contents",
            )
            record["content"] = content
            records.append(record)

        return records


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


def _build_metadata_record(
    *,
    context: SeedReaderFileSystemContext,
    relative_path: str,
    source_kind: str,
) -> dict[str, str]:
    return {
        "source_kind": source_kind,
        "source_path": str(context.root_path / relative_path),
        "relative_path": relative_path,
        "file_name": PurePosixPath(relative_path).name,
    }


def _normalize_relative_path(path: str) -> str:
    return path.lstrip("/")
