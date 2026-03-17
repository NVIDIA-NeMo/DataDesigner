# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.seed import IndexRange
from data_designer.config.seed_source import DirectorySeedSource, FileContentsSeedSource, LocalFileSeedSource
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.resources.seed_reader import (
    DataFrameSeedReader,
    DirectorySeedReader,
    FileContentsSeedReader,
    FileSystemSeedReader,
    LocalFileSeedReader,
    SeedReaderError,
    SeedReaderFileSystemContext,
    SeedReaderRegistry,
)
from data_designer.engine.secret_resolver import PlaintextResolver


class TrackingFileContentsSeedReader(FileContentsSeedReader):
    def __init__(self) -> None:
        self.hydrated_relative_paths: list[str] = []

    def hydrate_row(
        self,
        *,
        manifest_row: dict[str, str],
        context: SeedReaderFileSystemContext,
    ) -> dict[str, str]:
        self.hydrated_relative_paths.append(manifest_row["relative_path"])
        return super().hydrate_row(manifest_row=manifest_row, context=context)


class PluginStyleDirectorySeedReader(FileSystemSeedReader[DirectorySeedSource]):
    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> lazy.pd.DataFrame | list[dict[str, str]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [
            {
                "relative_path": relative_path,
                "file_name": Path(relative_path).name,
            }
            for relative_path in matched_paths
        ]


class CountingDataFrameSeedReader(DataFrameSeedReader):
    def __init__(self) -> None:
        self.create_duckdb_connection_calls = 0

    def create_duckdb_connection(self) -> lazy.duckdb.DuckDBPyConnection:
        self.create_duckdb_connection_calls += 1
        return super().create_duckdb_connection()


class OnAttachDirectorySeedReader(FileSystemSeedReader[DirectorySeedSource]):
    def __init__(self, label_prefix: str) -> None:
        self.label_prefix = label_prefix
        self.attach_call_count = 0

    def on_attach(self) -> None:
        self.attach_call_count += 1

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> lazy.pd.DataFrame | list[dict[str, str]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [
            {
                "relative_path": relative_path,
                "label": f"{self.label_prefix}:{Path(relative_path).name}",
            }
            for relative_path in matched_paths
        ]


class UndeclaredHydrationColumnSeedReader(FileSystemSeedReader[DirectorySeedSource]):
    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> lazy.pd.DataFrame | list[dict[str, str]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [{"relative_path": relative_path} for relative_path in matched_paths]

    def hydrate_row(
        self,
        *,
        manifest_row: dict[str, str],
        context: SeedReaderFileSystemContext,
    ) -> dict[str, str]:
        hydrated_row = dict(manifest_row)
        hydrated_row["content"] = str(context.root_path / manifest_row["relative_path"])
        return hydrated_row


class MissingHydrationColumnSeedReader(FileSystemSeedReader[DirectorySeedSource]):
    output_columns = ["relative_path", "content"]

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> lazy.pd.DataFrame | list[dict[str, str]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [{"relative_path": relative_path} for relative_path in matched_paths]

    def hydrate_row(
        self,
        *,
        manifest_row: dict[str, str],
        context: SeedReaderFileSystemContext,
    ) -> dict[str, str]:
        if manifest_row["relative_path"] == "beta.txt":
            return {
                "relative_path": manifest_row["relative_path"],
                "content": str(context.root_path / manifest_row["relative_path"]),
            }
        return {
            "relative_path": manifest_row["relative_path"],
        }


class FanoutDirectorySeedReader(FileSystemSeedReader[DirectorySeedSource]):
    output_columns = ["relative_path", "file_name", "line_index", "line"]

    def __init__(self) -> None:
        self.hydrated_relative_paths: list[str] = []

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> lazy.pd.DataFrame | list[dict[str, str]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [
            {
                "relative_path": relative_path,
                "file_name": Path(relative_path).name,
            }
            for relative_path in matched_paths
        ]

    def hydrate_row(
        self,
        *,
        manifest_row: dict[str, str],
        context: SeedReaderFileSystemContext,
    ) -> list[dict[str, Any]]:
        relative_path = manifest_row["relative_path"]
        self.hydrated_relative_paths.append(relative_path)
        with context.fs.open(relative_path, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        return [
            {
                "relative_path": relative_path,
                "file_name": manifest_row["file_name"],
                "line_index": line_index,
                "line": line,
            }
            for line_index, line in enumerate(lines)
        ]


class InvalidHydrationReturnSeedReader(FileSystemSeedReader[DirectorySeedSource]):
    def __init__(self, hydrated_return: Any) -> None:
        self._hydrated_return = hydrated_return

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> lazy.pd.DataFrame | list[dict[str, str]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [{"relative_path": relative_path} for relative_path in matched_paths]

    def hydrate_row(
        self,
        *,
        manifest_row: dict[str, str],
        context: SeedReaderFileSystemContext,
    ) -> Any:
        del manifest_row, context
        return self._hydrated_return


class SchemaMismatchFanoutSeedReader(FileSystemSeedReader[DirectorySeedSource]):
    def __init__(self, *, output_columns: list[str], hydrated_rows: list[dict[str, str]]) -> None:
        self.output_columns = output_columns
        self._hydrated_rows = hydrated_rows

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> lazy.pd.DataFrame | list[dict[str, str]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [{"relative_path": relative_path} for relative_path in matched_paths]

    def hydrate_row(
        self,
        *,
        manifest_row: dict[str, str],
        context: SeedReaderFileSystemContext,
    ) -> list[dict[str, str]]:
        del manifest_row, context
        return self._hydrated_rows


class ContextCountingDirectorySeedReader(FileSystemSeedReader[DirectorySeedSource]):
    def __init__(self) -> None:
        self.filesystem_context_calls = 0

    def create_filesystem_context(self, root_path: Path | str) -> SeedReaderFileSystemContext:
        self.filesystem_context_calls += 1
        return super().create_filesystem_context(root_path)

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> lazy.pd.DataFrame | list[dict[str, str]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [{"relative_path": relative_path} for relative_path in matched_paths]


def test_one_reader_per_seed_type():
    local_1 = LocalFileSeedReader()
    local_2 = LocalFileSeedReader()

    with pytest.raises(SeedReaderError):
        SeedReaderRegistry([local_1, local_2])

    registry = SeedReaderRegistry([local_1])

    with pytest.raises(SeedReaderError):
        registry.add_reader(local_2)


def test_get_reader_basic():
    local_reader = LocalFileSeedReader()
    df_reader = DataFrameSeedReader()
    registry = SeedReaderRegistry([local_reader, df_reader])

    df = lazy.pd.DataFrame(data={"a": [1, 2, 3]})
    local_seed_config = DataFrameSeedSource(df=df)

    reader = registry.get_reader(local_seed_config, PlaintextResolver())

    assert reader == df_reader


def test_get_reader_missing():
    local_reader = LocalFileSeedReader()
    registry = SeedReaderRegistry([local_reader])

    df = lazy.pd.DataFrame(data={"a": [1, 2, 3]})
    local_seed_config = DataFrameSeedSource(df=df)

    with pytest.raises(SeedReaderError):
        registry.get_reader(local_seed_config, PlaintextResolver())


def test_filesystem_seed_readers_expose_seed_type() -> None:
    assert DirectorySeedReader().get_seed_type() == "directory"
    assert FileContentsSeedReader().get_seed_type() == "file_contents"


def test_seed_reader_requires_attach_before_use() -> None:
    reader = DataFrameSeedReader()

    with pytest.raises(SeedReaderError, match="must be attached to a source"):
        reader.get_seed_dataset_size()


def test_plugin_style_filesystem_seed_reader_needs_only_manifest_builder(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "beta.txt").write_text("beta", encoding="utf-8")

    reader = PluginStyleDirectorySeedReader()
    reader.attach(
        DirectorySeedSource(path=str(tmp_path), file_pattern="*.txt"),
        PlaintextResolver(),
    )

    df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()

    assert reader.get_dataset_uri() == "seed_reader_directory_rows"
    assert list(df["relative_path"]) == ["alpha.txt", "nested/beta.txt"]
    assert list(df["file_name"]) == ["alpha.txt", "beta.txt"]


def test_plugin_style_filesystem_seed_reader_can_fan_out_rows(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha-0\nalpha-1", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("beta-0", encoding="utf-8")

    reader = FanoutDirectorySeedReader()
    reader.attach(
        DirectorySeedSource(path=str(tmp_path), file_pattern="*.txt"),
        PlaintextResolver(),
    )

    df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()

    assert reader.get_seed_dataset_size() == 2
    assert list(df["relative_path"]) == ["alpha.txt", "alpha.txt", "beta.txt"]
    assert list(df["line_index"]) == [0, 1, 0]
    assert list(df["line"]) == ["alpha-0", "alpha-1", "beta-0"]


def test_directory_seed_reader_matches_files_recursively(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "beta.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "nested" / "gamma.md").write_text("gamma", encoding="utf-8")

    reader = DirectorySeedReader()
    reader.attach(
        DirectorySeedSource(path=str(tmp_path), file_pattern="*.txt"),
        PlaintextResolver(),
    )

    df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()

    assert list(df["relative_path"]) == ["alpha.txt", "nested/beta.txt"]
    assert list(df["source_kind"]) == ["directory_file", "directory_file"]
    assert list(df["file_name"]) == ["alpha.txt", "beta.txt"]


def test_directory_seed_reader_can_disable_recursive_walk(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "beta.txt").write_text("beta", encoding="utf-8")

    reader = DirectorySeedReader()
    reader.attach(
        DirectorySeedSource(path=str(tmp_path), file_pattern="*.txt", recursive=False),
        PlaintextResolver(),
    )

    df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()

    assert list(df["relative_path"]) == ["alpha.txt"]


def test_directory_seed_reader_raises_for_no_matches(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")

    reader = DirectorySeedReader()
    reader.attach(
        DirectorySeedSource(path=str(tmp_path), file_pattern="*.md"),
        PlaintextResolver(),
    )

    with pytest.raises(SeedReaderError, match="No files matched file_pattern '\\*\\.md'"):
        reader.get_seed_dataset_size()


def test_file_contents_seed_reader_reads_text_files(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "beta.txt").write_text("beta", encoding="utf-8")

    reader = FileContentsSeedReader()
    reader.attach(
        FileContentsSeedSource(path=str(tmp_path), file_pattern="*.txt"),
        PlaintextResolver(),
    )

    df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()

    assert list(df["relative_path"]) == ["alpha.txt", "nested/beta.txt"]
    assert list(df["content"]) == ["alpha", "beta"]
    assert list(df["source_kind"]) == ["file_contents", "file_contents"]


def test_file_contents_seed_reader_respects_encoding(tmp_path: Path) -> None:
    file_path = tmp_path / "latin1.txt"
    file_path.write_bytes("café".encode("latin-1"))

    reader = FileContentsSeedReader()
    reader.attach(
        FileContentsSeedSource(path=str(tmp_path), file_pattern="*.txt", encoding="latin-1"),
        PlaintextResolver(),
    )

    df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()

    assert list(df["content"]) == ["café"]


def test_file_contents_seed_reader_wraps_unknown_encoding_errors(tmp_path: Path) -> None:
    file_path = tmp_path / "alpha.txt"
    file_path.write_text("alpha", encoding="utf-8")

    source = FileContentsSeedSource.model_construct(
        seed_type="file_contents",
        path=str(tmp_path),
        file_pattern="*.txt",
        recursive=True,
        encoding="utf-999",
    )
    reader = FileContentsSeedReader()
    reader.attach(source, PlaintextResolver())

    with pytest.raises(SeedReaderError, match="Failed to decode file .* using encoding 'utf-999'"):
        reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()


def test_file_contents_seed_reader_hydrates_only_selected_manifest_rows(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "gamma.txt").write_text("gamma", encoding="utf-8")

    reader = TrackingFileContentsSeedReader()
    reader.attach(
        FileContentsSeedSource(path=str(tmp_path), file_pattern="*.txt"),
        PlaintextResolver(),
    )

    batch_reader = reader.create_batch_reader(
        batch_size=1,
        index_range=IndexRange(start=1, end=1),
        shuffle=False,
    )
    batch_df = batch_reader.read_next_batch().to_pandas()

    assert list(batch_df["relative_path"]) == ["beta.txt"]
    assert list(batch_df["content"]) == ["beta"]
    assert reader.hydrated_relative_paths == ["beta.txt"]


def test_filesystem_seed_reader_fanout_keeps_manifest_based_index_selection(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha-0\nalpha-1", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("beta-0\nbeta-1", encoding="utf-8")

    reader = FanoutDirectorySeedReader()
    reader.attach(
        DirectorySeedSource(path=str(tmp_path), file_pattern="*.txt"),
        PlaintextResolver(),
    )

    batch_reader = reader.create_batch_reader(
        batch_size=1,
        index_range=IndexRange(start=1, end=1),
        shuffle=False,
    )
    batch_df = batch_reader.read_next_batch().to_pandas()

    assert list(batch_df["relative_path"]) == ["beta.txt", "beta.txt"]
    assert list(batch_df["line"]) == ["beta-0", "beta-1"]
    assert reader.hydrated_relative_paths == ["beta.txt"]


def test_filesystem_seed_reader_batch_reader_raises_for_selected_manifest_rows_with_empty_fanout(
    tmp_path: Path,
) -> None:
    (tmp_path / "alpha.txt").write_text("alpha-0", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("", encoding="utf-8")

    reader = FanoutDirectorySeedReader()
    reader.attach(
        DirectorySeedSource(path=str(tmp_path), file_pattern="*.txt"),
        PlaintextResolver(),
    )

    batch_reader = reader.create_batch_reader(
        batch_size=1,
        index_range=IndexRange(start=1, end=1),
        shuffle=False,
    )

    with pytest.raises(
        SeedReaderError,
        match="Selected manifest rows for seed source at .* did not produce any rows after hydration",
    ):
        batch_reader.read_next_batch()

    assert reader.hydrated_relative_paths == ["beta.txt"]


def test_filesystem_seed_reader_batch_reader_skips_empty_fanout_rows_before_returning_records(
    tmp_path: Path,
) -> None:
    (tmp_path / "alpha.txt").write_text("", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("beta-0\nbeta-1", encoding="utf-8")

    reader = FanoutDirectorySeedReader()
    reader.attach(
        DirectorySeedSource(path=str(tmp_path), file_pattern="*.txt"),
        PlaintextResolver(),
    )

    batch_reader = reader.create_batch_reader(
        batch_size=1,
        index_range=None,
        shuffle=False,
    )
    batch_df = batch_reader.read_next_batch().to_pandas()

    assert list(batch_df["relative_path"]) == ["beta.txt", "beta.txt"]
    assert list(batch_df["line"]) == ["beta-0", "beta-1"]
    assert reader.hydrated_relative_paths == ["alpha.txt", "beta.txt"]


def test_filesystem_seed_reader_batch_reader_stops_cleanly_after_emitting_records_when_only_empty_fanout_rows_remain(
    tmp_path: Path,
) -> None:
    (tmp_path / "alpha.txt").write_text("alpha-0", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("", encoding="utf-8")

    reader = FanoutDirectorySeedReader()
    reader.attach(
        DirectorySeedSource(path=str(tmp_path), file_pattern="*.txt"),
        PlaintextResolver(),
    )

    batch_reader = reader.create_batch_reader(
        batch_size=1,
        index_range=None,
        shuffle=False,
    )
    batch_df = batch_reader.read_next_batch().to_pandas()

    assert list(batch_df["relative_path"]) == ["alpha.txt"]
    assert list(batch_df["line"]) == ["alpha-0"]

    with pytest.raises(StopIteration):
        batch_reader.read_next_batch()

    assert reader.hydrated_relative_paths == ["alpha.txt", "beta.txt"]


def test_local_file_seed_reader_uses_load_time_runtime_path_when_cwd_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_root = tmp_path / "initial"
    later_root = tmp_path / "later"
    initial_root.mkdir()
    later_root.mkdir()

    lazy.pd.DataFrame({"value": [1]}).to_parquet(initial_root / "seed.parquet", index=False)
    lazy.pd.DataFrame({"value": [2]}).to_parquet(later_root / "seed.parquet", index=False)

    monkeypatch.chdir(initial_root)
    source = LocalFileSeedSource(path="seed.parquet")
    reader = LocalFileSeedReader()

    monkeypatch.chdir(later_root)
    reader.attach(source, PlaintextResolver())
    df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()

    assert source.path == "seed.parquet"
    assert reader.get_dataset_uri() == str((initial_root / "seed.parquet").resolve())
    assert list(df["value"]) == [1]


def test_directory_seed_reader_uses_load_time_runtime_path_when_cwd_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_root = tmp_path / "initial"
    later_root = tmp_path / "later"
    initial_seed_dir = initial_root / "seed-dir"
    later_seed_dir = later_root / "seed-dir"
    initial_seed_dir.mkdir(parents=True)
    later_seed_dir.mkdir(parents=True)
    (initial_seed_dir / "alpha.txt").write_text("alpha", encoding="utf-8")
    (later_seed_dir / "beta.txt").write_text("beta", encoding="utf-8")

    monkeypatch.chdir(initial_root)
    source = DirectorySeedSource(path="seed-dir", file_pattern="*.txt")
    reader = DirectorySeedReader()

    monkeypatch.chdir(later_root)
    reader.attach(source, PlaintextResolver())
    df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()

    assert source.path == "seed-dir"
    assert list(df["relative_path"]) == ["alpha.txt"]
    assert list(df["source_path"]) == [str((initial_seed_dir / "alpha.txt").resolve())]


def test_filesystem_seed_reader_on_attach_requires_no_super_and_resets_state(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    first_dir.mkdir()
    (first_dir / "alpha.txt").write_text("alpha", encoding="utf-8")
    (first_dir / "beta.txt").write_text("beta", encoding="utf-8")

    second_dir = tmp_path / "second"
    second_dir.mkdir()
    (second_dir / "gamma.txt").write_text("gamma", encoding="utf-8")

    reader = OnAttachDirectorySeedReader(label_prefix="plugin")

    reader.attach(DirectorySeedSource(path=str(first_dir), file_pattern="*.txt"), PlaintextResolver())
    first_df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()

    assert reader.attach_call_count == 1
    assert reader.get_seed_dataset_size() == 2
    assert list(first_df["label"]) == ["plugin:alpha.txt", "plugin:beta.txt"]

    reader.attach(DirectorySeedSource(path=str(second_dir), file_pattern="*.txt"), PlaintextResolver())
    second_df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()

    assert reader.attach_call_count == 2
    assert reader.get_seed_dataset_size() == 1
    assert list(second_df["relative_path"]) == ["gamma.txt"]
    assert list(second_df["label"]) == ["plugin:gamma.txt"]


@pytest.mark.parametrize("use_batch_reader", [False, True], ids=["full-output", "batch-reader"])
def test_filesystem_seed_reader_raises_for_undeclared_hydrated_columns(
    tmp_path: Path,
    use_batch_reader: bool,
) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")

    reader = UndeclaredHydrationColumnSeedReader()
    reader.attach(DirectorySeedSource(path=str(tmp_path), file_pattern="*.txt"), PlaintextResolver())

    with pytest.raises(SeedReaderError, match="Undeclared columns: \\['content'\\]"):
        if use_batch_reader:
            batch_reader = reader.create_batch_reader(
                batch_size=1,
                index_range=IndexRange(start=0, end=0),
                shuffle=False,
            )
            batch_reader.read_next_batch().to_pandas()
        else:
            reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()


@pytest.mark.parametrize(
    ("hydrated_return", "error_pattern"),
    [
        (123, "Manifest row index 0 returned int"),
        (["not-a-record"], "Manifest row index 0 returned an iterable containing str"),
    ],
    ids=["scalar", "iterable-of-invalid-records"],
)
def test_filesystem_seed_reader_rejects_invalid_hydrate_row_returns(
    tmp_path: Path,
    hydrated_return: Any,
    error_pattern: str,
) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")

    reader = InvalidHydrationReturnSeedReader(hydrated_return)
    reader.attach(DirectorySeedSource(path=str(tmp_path), file_pattern="*.txt"), PlaintextResolver())

    with pytest.raises(SeedReaderError, match=error_pattern):
        reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()


@pytest.mark.parametrize(
    ("output_columns", "hydrated_rows", "error_pattern"),
    [
        (
            ["relative_path", "content"],
            [
                {"relative_path": "alpha.txt", "content": "alpha"},
                {"relative_path": "alpha.txt"},
            ],
            "Hydrated record at index 1 .* Missing columns: \\['content'\\]",
        ),
        (
            ["relative_path"],
            [
                {"relative_path": "alpha.txt"},
                {"relative_path": "alpha.txt", "content": "alpha"},
            ],
            "Hydrated record at index 1 .* Undeclared columns: \\['content'\\]",
        ),
    ],
    ids=["missing-column", "undeclared-column"],
)
def test_filesystem_seed_reader_validates_each_fanout_record_against_output_columns(
    tmp_path: Path,
    output_columns: list[str],
    hydrated_rows: list[dict[str, str]],
    error_pattern: str,
) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")

    reader = SchemaMismatchFanoutSeedReader(output_columns=output_columns, hydrated_rows=hydrated_rows)
    reader.attach(DirectorySeedSource(path=str(tmp_path), file_pattern="*.txt"), PlaintextResolver())

    with pytest.raises(SeedReaderError, match=error_pattern):
        reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()


@pytest.mark.parametrize("use_batch_reader", [False, True], ids=["full-output", "batch-reader"])
def test_filesystem_seed_reader_raises_for_missing_declared_hydrated_columns(
    tmp_path: Path,
    use_batch_reader: bool,
) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("beta", encoding="utf-8")

    reader = MissingHydrationColumnSeedReader()
    reader.attach(DirectorySeedSource(path=str(tmp_path), file_pattern="*.txt"), PlaintextResolver())

    with pytest.raises(SeedReaderError, match="Missing columns: \\['content'\\]"):
        if use_batch_reader:
            batch_reader = reader.create_batch_reader(
                batch_size=2,
                index_range=IndexRange(start=0, end=1),
                shuffle=False,
            )
            batch_reader.read_next_batch().to_pandas()
        else:
            reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()


def test_filesystem_seed_reader_reuses_filesystem_context_until_reattach(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    first_dir.mkdir()
    (first_dir / "alpha.txt").write_text("alpha", encoding="utf-8")
    (first_dir / "beta.txt").write_text("beta", encoding="utf-8")

    second_dir = tmp_path / "second"
    second_dir.mkdir()
    (second_dir / "gamma.txt").write_text("gamma", encoding="utf-8")

    reader = ContextCountingDirectorySeedReader()

    reader.attach(DirectorySeedSource(path=str(first_dir), file_pattern="*.txt"), PlaintextResolver())

    assert reader.get_seed_dataset_size() == 2
    assert reader.filesystem_context_calls == 1

    batch_reader = reader.create_batch_reader(
        batch_size=1,
        index_range=IndexRange(start=0, end=0),
        shuffle=False,
    )
    assert list(batch_reader.read_next_batch().to_pandas()["relative_path"]) == ["alpha.txt"]
    assert reader.filesystem_context_calls == 1

    first_df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()
    assert list(first_df["relative_path"]) == ["alpha.txt", "beta.txt"]
    assert reader.filesystem_context_calls == 1

    reader.attach(DirectorySeedSource(path=str(second_dir), file_pattern="*.txt"), PlaintextResolver())

    assert reader.get_seed_dataset_size() == 1
    assert reader.filesystem_context_calls == 2


def test_seed_reader_reuses_cached_duckdb_connection_until_reattach() -> None:
    reader = CountingDataFrameSeedReader()
    reader.attach(DataFrameSeedSource(df=lazy.pd.DataFrame({"value": [1, 2, 3]})), PlaintextResolver())

    assert reader.get_seed_dataset_size() == 3
    assert reader.get_column_names() == ["value"]
    batch_reader = reader.create_batch_reader(
        batch_size=2,
        index_range=IndexRange(start=0, end=1),
        shuffle=False,
    )

    assert list(batch_reader.read_next_batch().to_pandas()["value"]) == [1, 2]
    assert reader.create_duckdb_connection_calls == 1

    reader.attach(DataFrameSeedSource(df=lazy.pd.DataFrame({"value": [9]})), PlaintextResolver())

    assert reader.get_seed_dataset_size() == 1
    assert reader.create_duckdb_connection_calls == 2
