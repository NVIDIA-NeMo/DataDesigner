# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar, get_args, get_origin

from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from huggingface_hub import HfFileSystem
from typing_extensions import Self

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.seed import IndexRange
from data_designer.config.seed_source import (
    AgentRolloutSeedSource,
    DirectorySeedSource,
    FileContentsSeedSource,
    FileSystemSeedSource,
    HuggingFaceSeedSource,
    LocalFileSeedSource,
    SeedSource,
)
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.resources.agent_rollout import (
    AgentRolloutFormatHandler,
    AgentRolloutParseContext,
    AgentRolloutSeedParseError,
    NormalizedAgentRolloutRecord,
    get_format_handler,
)
from data_designer.engine.secret_resolver import SecretResolver
from data_designer.errors import DataDesignerError

if TYPE_CHECKING:
    import duckdb
    import pandas as pd

logger = logging.getLogger(__name__)


class SeedReaderError(DataDesignerError): ...


@dataclass(frozen=True)
class SeedReaderFileSystemContext:
    fs: AbstractFileSystem
    root_path: Path


class SeedReaderBatch(Protocol):
    def to_pandas(self) -> pd.DataFrame: ...


class SeedReaderBatchReader(Protocol):
    def read_next_batch(self) -> SeedReaderBatch: ...


@dataclass
class PandasSeedReaderBatch:
    dataframe: pd.DataFrame

    def to_pandas(self) -> pd.DataFrame:
        return self.dataframe


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
            f"Hydrated record at index {row_index} does not match output_columns {output_columns!r}."
        ]
        if missing_columns:
            message_parts.append(f"Missing columns: {missing_columns!r}.")
        if extra_columns:
            message_parts.append(f"Undeclared columns: {extra_columns!r}.")
        message_parts.append("Ensure each record emitted by hydrate_row() matches the declared output schema.")
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

    def read_next_batch(self) -> SeedReaderBatch:
        return self._batch_reader.read_next_batch()


class HydratingSeedReaderBatchReader:
    def __init__(
        self,
        *,
        manifest_batch_reader: SeedReaderBatchReader,
        hydrate_records: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
        output_columns: list[str],
        no_rows_error_message: str,
    ) -> None:
        self._manifest_batch_reader = manifest_batch_reader
        self._hydrate_records = hydrate_records
        self._output_columns = output_columns
        self._no_rows_error_message = no_rows_error_message
        self._has_emitted_records = False

    def read_next_batch(self) -> SeedReaderBatch:
        while True:
            try:
                manifest_batch = self._manifest_batch_reader.read_next_batch()
            except StopIteration:
                if self._has_emitted_records:
                    raise
                raise SeedReaderError(self._no_rows_error_message) from None

            manifest_records = manifest_batch.to_pandas().to_dict(orient="records")
            hydrated_records = self._hydrate_records(manifest_records)
            if not hydrated_records:
                continue

            self._has_emitted_records = True
            return PandasSeedReaderBatch(
                create_seed_reader_output_dataframe(records=hydrated_records, output_columns=self._output_columns)
            )


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
        return self.source.runtime_path


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
    expensive enrichment can optionally override `hydrate_row(...)` to emit one
    record dict or an iterable of record dicts per manifest row. When emitted
    records change the manifest schema, `output_columns` must declare the exact
    hydrated output schema for each emitted record. The framework owns
    attachment-scoped filesystem context reuse, manifest sampling, partitioning,
    randomization, batching, and DuckDB registration details.
    """

    output_columns: ClassVar[list[str] | None] = None

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
    ) -> dict[str, Any] | Iterable[dict[str, Any]]:
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
        context = self._get_filesystem_context()
        conn = self.create_dataframe_duckdb_connection(
            table_name=self._get_manifest_dataset_uri(),
            dataframe=self._get_row_manifest_dataframe(),
        )
        read_query = self.build_dataset_read_query(
            dataset_uri=self._get_manifest_dataset_uri(),
            index_range=index_range,
            shuffle=shuffle,
        )
        query_result = conn.query(read_query)
        manifest_batch_reader = DuckDBSeedReaderBatchReader(conn=conn, query_result=query_result, batch_size=batch_size)
        return HydratingSeedReaderBatchReader(
            manifest_batch_reader=manifest_batch_reader,
            hydrate_records=lambda manifest_records: self._hydrate_rows(
                manifest_rows=manifest_records,
                context=context,
            ),
            output_columns=self.get_output_column_names(),
            no_rows_error_message=self._get_empty_selected_manifest_rows_error_message(),
        )

    def _get_row_manifest_dataframe(self) -> pd.DataFrame:
        self._ensure_attached()
        manifest_df = getattr(self, "_row_manifest_df", None)
        if manifest_df is not None:
            return manifest_df

        context = self._get_filesystem_context()
        manifest = self.build_manifest(context=context)
        manifest_df = self._normalize_rows_to_dataframe(manifest)
        if manifest_df.empty:
            raise SeedReaderError(f"Seed source at {self.source.runtime_path} did not produce any rows")

        self._row_manifest_df = manifest_df
        return self._row_manifest_df

    def _get_output_dataframe(self) -> pd.DataFrame:
        self._ensure_attached()
        output_df = getattr(self, "_output_df", None)
        if output_df is not None:
            return output_df

        context = self._get_filesystem_context()
        hydrated_records = self._hydrate_rows(
            manifest_rows=self._get_row_manifest_dataframe().to_dict(orient="records"),
            context=context,
        )
        if not hydrated_records:
            raise SeedReaderError(f"Seed source at {self.source.runtime_path} did not produce any rows")

        self._output_df = create_seed_reader_output_dataframe(
            records=hydrated_records,
            output_columns=self.get_output_column_names(),
        )
        return self._output_df

    def _get_filesystem_context(self) -> SeedReaderFileSystemContext:
        self._ensure_attached()
        context = getattr(self, "_filesystem_context", None)
        if context is None:
            context = self.create_filesystem_context(self.source.runtime_path)
            self._filesystem_context = context
        return context

    def _get_manifest_dataset_uri(self) -> str:
        return self._build_internal_table_name("manifest")

    def _build_internal_table_name(self, suffix: str) -> str:
        seed_type = self.get_seed_type().replace("-", "_")
        return f"seed_reader_{seed_type}_{suffix}"

    def _get_empty_selected_manifest_rows_error_message(self) -> str:
        return f"Selected manifest rows for seed source at {self.source.runtime_path} did not produce any rows after hydration"

    def _normalize_rows_to_dataframe(self, rows: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
        if isinstance(rows, lazy.pd.DataFrame):
            return rows.copy()
        return lazy.pd.DataFrame(rows)

    def _hydrate_rows(
        self,
        *,
        manifest_rows: list[dict[str, Any]],
        context: SeedReaderFileSystemContext,
    ) -> list[dict[str, Any]]:
        hydrated_records: list[dict[str, Any]] = []
        for manifest_row_index, manifest_row in enumerate(manifest_rows):
            hydrated_records.extend(
                _normalize_hydrated_row_output(
                    hydrated_row_output=self.hydrate_row(manifest_row=manifest_row, context=context),
                    manifest_row_index=manifest_row_index,
                )
            )
        return hydrated_records


class DirectorySeedReader(FileSystemSeedReader[DirectorySeedSource]):
    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> pd.DataFrame | list[dict[str, Any]]:
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
    output_columns = ["source_kind", "source_path", "relative_path", "file_name", "content"]

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> pd.DataFrame | list[dict[str, Any]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [
            _build_metadata_record(
                context=context,
                relative_path=relative_path,
                source_kind="file_contents",
            )
            for relative_path in matched_paths
        ]

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


class AgentRolloutSeedReader(FileSystemSeedReader[AgentRolloutSeedSource]):
    output_columns = NormalizedAgentRolloutRecord.get_field_names()

    _PARSE_CONTEXT_UNSET: AgentRolloutParseContext | None = object()  # type: ignore[assignment]

    def _reset_attachment_state(self) -> None:
        super()._reset_attachment_state()
        self._parse_context: AgentRolloutParseContext | None = self._PARSE_CONTEXT_UNSET  # type: ignore[assignment]

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> list[dict[str, Any]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.resolved_file_pattern,
            recursive=self.source.recursive,
        )
        handler = self.get_format_handler()
        handled: list[str] = []
        for p in matched_paths:
            if handler.is_handled_file(p):
                handled.append(p)
            elif handler.should_warn_unhandled_file(p):
                logger.warning("Skipping unhandled %s file %s", self.source.format.value, p)
        return [
            _build_metadata_record(context=context, relative_path=p, source_kind=self.source.format.value)
            for p in handled
        ]

    def hydrate_row(
        self,
        *,
        manifest_row: dict[str, Any],
        context: SeedReaderFileSystemContext,
    ) -> list[dict[str, Any]]:
        handler = self.get_format_handler()
        relative_path = manifest_row["relative_path"]
        try:
            parse_ctx = self._get_parse_context(context)
            records = handler.parse_file(
                root_path=context.root_path,
                relative_path=relative_path,
                parse_context=parse_ctx,
            )
        except (AgentRolloutSeedParseError, UnicodeDecodeError) as error:
            logger.warning("Skipping malformed file %s: %s", relative_path, error)
            return []
        except OSError as error:
            raise SeedReaderError(
                f"Failed to read agent rollout file {context.root_path / relative_path}: {error}"
            ) from error
        return [r.to_dict() for r in records]

    def get_format_handler(self) -> AgentRolloutFormatHandler:
        rollout_format = self.source.format
        try:
            return get_format_handler(rollout_format)
        except KeyError as error:
            raise SeedReaderError(
                f"No AgentRollout format handler found for format {rollout_format.value!r}"
            ) from error

    def _get_parse_context(self, context: SeedReaderFileSystemContext) -> AgentRolloutParseContext | None:
        if self._parse_context is not self._PARSE_CONTEXT_UNSET:
            return self._parse_context

        handler = self.get_format_handler()
        self._parse_context = handler.build_parse_context(root_path=context.root_path, recursive=self.source.recursive)
        return self._parse_context


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


def _normalize_hydrated_row_output(
    *,
    hydrated_row_output: dict[str, Any] | Iterable[dict[str, Any]],
    manifest_row_index: int,
) -> list[dict[str, Any]]:
    if isinstance(hydrated_row_output, dict):
        return [hydrated_row_output]

    if not isinstance(hydrated_row_output, Iterable):
        raise SeedReaderError(
            "hydrate_row() must return a record dict or an iterable of record dicts. "
            f"Manifest row index {manifest_row_index} returned {type(hydrated_row_output).__name__}."
        )

    hydrated_records = list(hydrated_row_output)
    for hydrated_record in hydrated_records:
        if isinstance(hydrated_record, dict):
            continue
        raise SeedReaderError(
            "hydrate_row() must return a record dict or an iterable of record dicts. "
            f"Manifest row index {manifest_row_index} returned an iterable containing "
            f"{type(hydrated_record).__name__}."
        )

    return hydrated_records
