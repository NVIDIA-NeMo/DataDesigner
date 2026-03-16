# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

import data_designer.config as dd
import data_designer.lazy_heavy_imports as lazy
from data_designer.config.errors import InvalidFilePathError
from data_designer.config.seed_source import DirectorySeedSource, FileContentsSeedSource, LocalFileSeedSource
from data_designer.config.seed_source_dataframe import DataFrameSeedSource


def create_partitions_in_path(temp_dir: Path, extension: str, num_files: int = 2) -> Path:
    df = lazy.pd.DataFrame({"col": [1, 2, 3]})

    for i in range(num_files):
        file_path = temp_dir / f"partition_{i}.{extension}"
        if extension == "parquet":
            df.to_parquet(file_path)
        elif extension == "csv":
            df.to_csv(file_path, index=False)
        elif extension == "json":
            df.to_json(file_path, orient="records", lines=True)
        elif extension == "jsonl":
            df.to_json(file_path, orient="records", lines=True)
    return temp_dir


def test_local_seed_dataset_reference_validation(tmp_path: Path):
    with pytest.raises(InvalidFilePathError, match="🛑 Path test/dataset.parquet is not a file."):
        LocalFileSeedSource(path="test/dataset.parquet")

    # Should not raise an error when referencing supported extensions with wildcard pattern.
    create_partitions_in_path(tmp_path, "parquet")
    create_partitions_in_path(tmp_path, "csv")
    create_partitions_in_path(tmp_path, "json")
    create_partitions_in_path(tmp_path, "jsonl")

    test_cases = ["parquet", "csv", "json", "jsonl"]
    try:
        for extension in test_cases:
            config = LocalFileSeedSource(path=f"{tmp_path}/*.{extension}")
            assert config.path == f"{tmp_path}/*.{extension}"
    except Exception as e:
        pytest.fail(f"Expected no exception, but got {e}")


def test_local_seed_dataset_reference_validation_error(tmp_path: Path):
    create_partitions_in_path(tmp_path, "parquet")
    with pytest.raises(InvalidFilePathError, match="does not contain files of type 'csv'"):
        LocalFileSeedSource(path=f"{tmp_path}/*.csv")


def test_local_source_from_dataframe(tmp_path: Path):
    df = lazy.pd.DataFrame({"col": [1, 2, 3]})
    filepath = f"{tmp_path}/data.parquet"

    source = LocalFileSeedSource.from_dataframe(df, filepath)

    assert source.path == filepath
    lazy.pd.testing.assert_frame_equal(df, lazy.pd.read_parquet(filepath))


def test_dataframe_seed_source_serialization():
    """Test that DataFrameSeedSource excludes the DataFrame field during serialization."""
    df = lazy.pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    source = DataFrameSeedSource(df=df)

    # Test model_dump excludes the df field
    serialized = source.model_dump(mode="json")
    assert "df" not in serialized
    assert serialized == {"seed_type": "df"}


def test_directory_seed_source_requires_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("alpha", encoding="utf-8")

    with pytest.raises(InvalidFilePathError, match="is not a directory"):
        DirectorySeedSource(path=str(file_path))


def test_directory_seed_source_preserves_relative_path_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_dir = tmp_path / "seed-dir"
    seed_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    source = DirectorySeedSource(path="seed-dir")

    assert source.path == "seed-dir"
    assert source.model_dump(mode="json")["path"] == "seed-dir"
    assert source.file_pattern == "*"
    assert source.recursive is True


def test_file_contents_seed_source_defaults() -> None:
    source = FileContentsSeedSource(path=".", file_pattern="*.md", recursive=False)

    assert source.seed_type == "file_contents"
    assert source.file_pattern == "*.md"
    assert source.recursive is False
    assert source.encoding == "utf-8"


def test_file_contents_seed_source_preserves_relative_path_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_dir = tmp_path / "seed-dir"
    seed_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    source = FileContentsSeedSource(path="seed-dir", file_pattern="*.txt")

    assert source.path == "seed-dir"
    assert source.model_dump(mode="json")["path"] == "seed-dir"


def test_seed_source_path_descriptions_document_cwd_resolution() -> None:
    local_path_description = LocalFileSeedSource.model_json_schema()["properties"]["path"]["description"]
    directory_path_description = DirectorySeedSource.model_json_schema()["properties"]["path"]["description"]
    file_contents_path_description = FileContentsSeedSource.model_json_schema()["properties"]["path"]["description"]

    assert "current working directory" in local_path_description
    assert "config file location" in local_path_description
    assert "current working directory" in directory_path_description
    assert "config file location" in directory_path_description
    assert "current working directory" in file_contents_path_description
    assert "config file location" in file_contents_path_description


def test_seed_sources_are_exported_from_config_module(tmp_path: Path) -> None:
    directory_source = dd.DirectorySeedSource(path=str(tmp_path))
    file_contents_source = dd.FileContentsSeedSource(path=str(tmp_path), file_pattern="*.txt")

    assert directory_source.seed_type == "directory"
    assert file_contents_source.seed_type == "file_contents"


def test_file_contents_seed_source_parses_from_dict(tmp_path: Path) -> None:
    source = FileContentsSeedSource.model_validate(
        {
            "path": str(tmp_path),
            "file_pattern": "*.txt",
            "recursive": False,
            "encoding": "latin-1",
        }
    )

    assert source.file_pattern == "*.txt"
    assert source.recursive is False
    assert source.encoding == "latin-1"


@pytest.mark.parametrize(
    ("source_type", "file_pattern", "error_message"),
    [
        pytest.param(DirectorySeedSource, "", "non-empty string", id="directory-empty"),
        pytest.param(DirectorySeedSource, "subdir/*.txt", "match file names, not relative paths", id="directory-posix"),
        pytest.param(FileContentsSeedSource, "", "non-empty string", id="contents-empty"),
        pytest.param(
            FileContentsSeedSource,
            r"subdir\\*.txt",
            "match file names, not relative paths",
            id="contents-windows",
        ),
    ],
)
def test_filesystem_seed_sources_reject_path_like_file_patterns(
    source_type: type[DirectorySeedSource] | type[FileContentsSeedSource],
    file_pattern: str,
    error_message: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=error_message):
        source_type(path=str(tmp_path), file_pattern=file_pattern)
