# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

import data_designer.config as dd
import data_designer.lazy_heavy_imports as lazy
from data_designer.config.errors import InvalidFilePathError
from data_designer.config.seed_source import DirectoryListingTransform, DirectorySeedSource, LocalFileSeedSource
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


def test_directory_seed_source_allows_directory(tmp_path: Path) -> None:
    source = DirectorySeedSource(
        path=str(tmp_path),
        transform=DirectoryListingTransform(file_pattern="*.txt", recursive=False),
    )

    assert source.seed_type == "directory"
    assert source.path == str(tmp_path)
    assert isinstance(source.transform, DirectoryListingTransform)
    assert source.transform.file_pattern == "*.txt"
    assert source.transform.recursive is False


def test_directory_seed_source_is_exported_from_config_module(tmp_path: Path) -> None:
    source = dd.DirectorySeedSource(path=str(tmp_path), transform=dd.DirectoryListingTransform())

    assert source.seed_type == "directory"
    assert isinstance(source.transform, dd.DirectoryListingTransform)


def test_directory_seed_source_preserves_relative_path_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_dir = tmp_path / "seed-dir"
    seed_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    source = DirectorySeedSource(path="seed-dir")

    assert source.path == "seed-dir"
    assert source.model_dump(mode="json")["path"] == "seed-dir"


def test_seed_source_path_descriptions_document_cwd_resolution() -> None:
    local_path_description = LocalFileSeedSource.model_json_schema()["properties"]["path"]["description"]
    directory_path_description = DirectorySeedSource.model_json_schema()["properties"]["path"]["description"]

    assert "current working directory" in local_path_description
    assert "config file location" in local_path_description
    assert "current working directory" in directory_path_description
    assert "config file location" in directory_path_description


def test_directory_seed_source_parses_builtin_transform_from_dict(tmp_path: Path) -> None:
    source = DirectorySeedSource.model_validate(
        {
            "path": str(tmp_path),
            "transform": {
                "transform_type": "directory_listing",
                "file_pattern": "*.txt",
                "recursive": False,
            },
        }
    )

    assert isinstance(source.transform, DirectoryListingTransform)
    assert source.transform.file_pattern == "*.txt"
    assert source.transform.recursive is False


@pytest.mark.parametrize(
    ("file_pattern", "error_message"),
    [
        pytest.param("", "non-empty string", id="empty"),
        pytest.param("subdir/*.txt", "must match file names, not relative paths", id="posix-path"),
        pytest.param(r"subdir\\*.txt", "must match file names, not relative paths", id="windows-path"),
    ],
)
def test_directory_listing_transform_rejects_path_like_file_patterns(file_pattern: str, error_message: str) -> None:
    with pytest.raises(ValueError, match=error_message):
        DirectoryListingTransform(file_pattern=file_pattern)
