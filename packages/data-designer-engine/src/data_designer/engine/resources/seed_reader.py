# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, get_args, get_origin

from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from huggingface_hub import HfFileSystem
from typing_extensions import Self

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.seed import IndexRange
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


class SeedReaderBatchReader(Protocol):
    def read_next_batch(self) -> pd.DataFrame: ...


def create_seed_reader_output_dataframe(
    *,
    records: list[dict[str, Any]],
    output_columns: list[str],
) -> pd.DataFrame:
    if not records:
        return lazy.pd.DataFrame(records, columns=output_columns)

    expected_columns = set(output_columns)
    for row_index, record in enumerate(records):
        record_columns = set(record)
        extra_columns = sorted(record_columns - expected_columns)
        missing_columns = [column for column in output_columns if column not in record]
        if not extra_columns and not missing_columns:
            continue

        message_parts: list[str] = [
            f"Hydrated row at index {row_index} does not match output_columns {output_columns!r}."
        ]
        if missing_columns:
            message_parts.append(f"Missing columns: {missing_columns!r}.")
        if extra_columns:
            message_parts.append(f"Undeclared columns: {extra_columns!r}.")
        message_parts.append("Ensure hydrate_row() returns exactly the declared output schema.")
        raise SeedReaderError(" ".join(message_parts))

    return lazy.pd.DataFrame(records, columns=output_columns)


class DuckDBSeedReaderBatchReader:
    def __init__(
        self,
        *,
        conn: duckdb.DuckDBPyConnection,
        query_result: Any,
        batch_size: int,
    ) -> None:
        # Keep the connection and query result alive for the lifetime of the Arrow
        # batch reader. Dropping these references can invalidate in-memory tables
        # or query state before the reader has finished yielding batches.
        self._conn = conn
        self._query_result = query_result
        if hasattr(query_result, "to_arrow_reader"):
            self._batch_reader = query_result.to_arrow_reader(batch_size=batch_size)
        else:
            self._batch_reader = query_result.fetch_arrow_reader(batch_size=batch_size)

    def read_next_batch(self) -> pd.DataFrame:
        return self._batch_reader.read_next_batch().to_pandas()


class FileSystemSeedReaderBatchReader:
    def __init__(
        self,
        *,
        manifest_dataframe: pd.DataFrame,
        batch_size: int,
        hydrate_manifest_dataframe: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> None:
        self._manifest_dataframe = manifest_dataframe.reset_index(drop=True)
        self._batch_size = batch_size
        self._hydrate_manifest_dataframe = hydrate_manifest_dataframe
        self._next_row_index = 0

    def read_next_batch(self) -> pd.DataFrame:
        if self._next_row_index >= len(self._manifest_dataframe):
            raise StopIteration

        batch_df = self._manifest_dataframe.iloc[self._next_row_index : self._next_row_index + self._batch_size]
        self._next_row_index += self._batch_size
        return self._hydrate_manifest_dataframe(batch_df.reset_index(drop=True))


SourceT = TypeVar("SourceT", bound=SeedSource)
FileSystemSourceT = TypeVar("FileSystemSourceT", bound=FileSystemSeedSource)


class SeedReader(ABC, Generic[SourceT]):
    """Base class for reading a seed dataset.

    Seeds are read using duckdb. Reader implementations define duckdb connection setup details
    and how to get a URI that can be queried with duckdb (i.e. "... FROM <uri> ...").

    The Data Designer engine automatically supplies the appropriate SeedSource
    and a SecretResolver to use for any secret fields in the config via
    `attach(...)`. Subclasses that need per-attachment setup can override
    `on_attach(...)` without needing to call `super()`.
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
        self._reset_attachment_state()
        self.source = source
        self.secret_resolver = secret_resolver
        self.on_attach()

    def on_attach(self) -> None:
        """Hook for subclasses that need per-attachment setup."""

    def _reset_attachment_state(self) -> None:
        self._duckdb_conn = None

    def create_dataframe_duckdb_connection(
        self,
        *,
        table_name: str,
        dataframe: pd.DataFrame,
    ) -> duckdb.DuckDBPyConnection:
        conn = lazy.duckdb.connect()
        conn.register(table_name, dataframe)
        return conn

    def get_seed_dataset_size(self) -> int:
        self._ensure_attached()
        conn = self._get_duckdb_connection()
        return conn.execute(f"SELECT COUNT(*) FROM '{self.get_dataset_uri()}'").fetchone()[0]

    def create_batch_reader(
        self,
        *,
        batch_size: int,
        index_range: IndexRange | None,
        shuffle: bool,
    ) -> SeedReaderBatchReader:
        self._ensure_attached()
        conn = self._get_duckdb_connection()
        read_query = self.build_dataset_read_query(
            dataset_uri=self.get_dataset_uri(),
            index_range=index_range,
            shuffle=shuffle,
        )
        query_result = conn.query(read_query)
        return DuckDBSeedReaderBatchReader(conn=conn, query_result=query_result, batch_size=batch_size)

    def get_column_names(self) -> list[str]:
        """Returns the seed dataset's column names"""
        self._ensure_attached()
        conn = self._get_duckdb_connection()
        describe_query = f"DESCRIBE SELECT * FROM '{self.get_dataset_uri()}'"
        column_descriptions = conn.execute(describe_query).fetchall()
        return [col[0] for col in column_descriptions]

    def _get_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        self._ensure_attached()
        conn = getattr(self, "_duckdb_conn", None)
        if conn is None:
            conn = self.create_duckdb_connection()
            self._duckdb_conn = conn
        return conn

    def _ensure_attached(self) -> None:
        if not hasattr(self, "source") or not hasattr(self, "secret_resolver"):
            raise SeedReaderError("SeedReader must be attached to a source before use")

    @staticmethod
    def build_dataset_read_query(
        *,
        dataset_uri: str,
        index_range: IndexRange | None,
        shuffle: bool,
    ) -> str:
        shuffle_query = " ORDER BY RANDOM()" if shuffle else ""

        if index_range is not None:
            offset_value = index_range.start
            limit_value = index_range.end - index_range.start + 1
            read_query = f"""
                SELECT * FROM '{dataset_uri}'
                LIMIT {limit_value} OFFSET {offset_value}
            """
            return f"SELECT * FROM ({read_query}){shuffle_query}"

        return f"SELECT * FROM '{dataset_uri}'{shuffle_query}"

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
    """Base class for filesystem-derived seed readers.

    Plugin authors implement `build_manifest(...)` to describe the cheap logical
    rows available under the configured filesystem root. Readers that need
    expensive enrichment can optionally override `hydrate_row(...)`. When
    `hydrate_row(...)` changes the manifest schema, `output_columns` must declare
    the exact hydrated output schema. The framework owns attachment-scoped
    filesystem context reuse, manifest sampling, partitioning, randomization,
    batching, and DuckDB registration details.
    """

    output_columns: list[str] | None = None
    source_kind: str

    def _reset_attachment_state(self) -> None:
        super()._reset_attachment_state()
        self._filesystem_context = None
        self._output_df = None
        self._row_manifest_df = None

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        return self.create_dataframe_duckdb_connection(
            table_name=self.get_dataset_uri(),
            dataframe=self._get_output_dataframe(),
        )

    def get_dataset_uri(self) -> str:
        return self._build_internal_table_name("rows")

    def get_output_column_names(self) -> list[str]:
        if self.output_columns is not None:
            return self.output_columns
        return list(self._get_row_manifest_dataframe().columns)

    @abstractmethod
    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> pd.DataFrame | list[dict[str, Any]]: ...

    def hydrate_row(
        self,
        *,
        manifest_row: dict[str, Any],
        context: SeedReaderFileSystemContext,
    ) -> dict[str, Any]:
        return manifest_row

    def get_column_names(self) -> list[str]:
        return self.get_output_column_names()

    def get_seed_dataset_size(self) -> int:
        self._ensure_attached()
        return len(self._get_row_manifest_dataframe())

    def create_batch_reader(
        self,
        *,
        batch_size: int,
        index_range: IndexRange | None,
        shuffle: bool,
    ) -> SeedReaderBatchReader:
        self._ensure_attached()
        return FileSystemSeedReaderBatchReader(
            manifest_dataframe=self._select_manifest_dataframe(index_range=index_range, shuffle=shuffle),
            batch_size=batch_size,
            hydrate_manifest_dataframe=self._hydrate_manifest_dataframe,
        )

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
        # In fsspec, maxdepth=1 means files directly under the root
        # (depth 0 = the root itself, depth 1 = direct children).
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

    def _build_metadata_manifest(self, *, context: SeedReaderFileSystemContext) -> list[dict[str, str]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [
            _build_metadata_record(
                context=context,
                relative_path=relative_path,
                source_kind=self.source_kind,
            )
            for relative_path in matched_paths
        ]

    def _get_row_manifest_dataframe(self) -> pd.DataFrame:
        self._ensure_attached()
        manifest_df = getattr(self, "_row_manifest_df", None)
        if manifest_df is not None:
            return manifest_df

        context = self._get_filesystem_context()
        manifest = self.build_manifest(context=context)
        manifest_df = self._normalize_rows_to_dataframe(manifest)
        if manifest_df.empty:
            raise SeedReaderError(f"Seed source at {self.source.path} did not produce any rows")

        self._row_manifest_df = manifest_df
        return self._row_manifest_df

    def _get_output_dataframe(self) -> pd.DataFrame:
        self._ensure_attached()
        output_df = getattr(self, "_output_df", None)
        if output_df is not None:
            return output_df

        self._output_df = self._hydrate_manifest_dataframe(
            self._get_row_manifest_dataframe(),
            raise_on_empty=True,
        )
        return self._output_df

    def _get_filesystem_context(self) -> SeedReaderFileSystemContext:
        self._ensure_attached()
        context = getattr(self, "_filesystem_context", None)
        if context is None:
            context = self.create_filesystem_context(self.source.path)
            self._filesystem_context = context
        return context

    def _build_internal_table_name(self, suffix: str) -> str:
        seed_type = self.get_seed_type().replace("-", "_")
        return f"seed_reader_{seed_type}_{suffix}"

    def _normalize_rows_to_dataframe(self, rows: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
        if isinstance(rows, lazy.pd.DataFrame):
            return rows.copy()
        return lazy.pd.DataFrame(rows)

    def _select_manifest_dataframe(
        self,
        *,
        index_range: IndexRange | None,
        shuffle: bool,
    ) -> pd.DataFrame:
        manifest_df = self._get_row_manifest_dataframe()
        if index_range is not None:
            manifest_df = manifest_df.iloc[index_range.start : index_range.end + 1]
        if shuffle:
            manifest_df = manifest_df.sample(frac=1)
        return manifest_df.reset_index(drop=True)

    def _hydrate_manifest_dataframe(
        self,
        manifest_dataframe: pd.DataFrame,
        *,
        raise_on_empty: bool = False,
    ) -> pd.DataFrame:
        context = self._get_filesystem_context()
        hydrated_records = self._hydrate_rows(
            manifest_rows=manifest_dataframe.to_dict(orient="records"),
            context=context,
        )
        if raise_on_empty and not hydrated_records:
            raise SeedReaderError(f"Seed source at {self.source.path} did not produce any rows")
        return create_seed_reader_output_dataframe(
            records=hydrated_records,
            output_columns=self.get_output_column_names(),
        )

    def _hydrate_rows(
        self,
        *,
        manifest_rows: list[dict[str, Any]],
        context: SeedReaderFileSystemContext,
    ) -> list[dict[str, Any]]:
        return [self.hydrate_row(manifest_row=manifest_row, context=context) for manifest_row in manifest_rows]


class DirectorySeedReader(FileSystemSeedReader[DirectorySeedSource]):
    source_kind = "directory_file"

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> pd.DataFrame | list[dict[str, Any]]:
        return self._build_metadata_manifest(context=context)


class FileContentsSeedReader(FileSystemSeedReader[FileContentsSeedSource]):
    source_kind = "file_contents"
    output_columns = ["source_kind", "source_path", "relative_path", "file_name", "content"]

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> pd.DataFrame | list[dict[str, Any]]:
        return self._build_metadata_manifest(context=context)

    def hydrate_row(
        self,
        *,
        manifest_row: dict[str, Any],
        context: SeedReaderFileSystemContext,
    ) -> dict[str, Any]:
        relative_path = manifest_row["relative_path"]
        absolute_path = context.root_path / relative_path
        try:
            with context.fs.open(relative_path, "r", encoding=self.source.encoding) as handle:
                content = handle.read()
        except (UnicodeDecodeError, LookupError) as error:
            raise SeedReaderError(
                f"Failed to decode file {absolute_path} using encoding {self.source.encoding!r}: {error}"
            ) from error
        except OSError as error:
            raise SeedReaderError(f"Failed to read file {absolute_path}: {error}") from error

        hydrated_record = dict(manifest_row)
        hydrated_record["content"] = content
        return hydrated_record


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
