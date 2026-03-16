# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.seed_source import DirectorySeedSource, FileContentsSeedSource
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.resources.seed_reader import (
    DataFrameSeedReader,
    DirectorySeedReader,
    FileContentsSeedReader,
    LocalFileSeedReader,
    SeedReaderError,
    SeedReaderRegistry,
)
from data_designer.engine.secret_resolver import PlaintextResolver


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
        reader.get_column_names()


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
