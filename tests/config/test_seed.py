# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Union
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_designer.config.errors import InvalidFileFormatError, InvalidFilePathError
from data_designer.config.seed import HfHubSeedDatasetReference, IndexRange, LocalSeedDatasetReference, PartitionBlock


def create_partitions_in_path(temp_dir: Path, extension: str, num_files: int = 2) -> Path:
    df = pd.DataFrame({"col": [1, 2, 3]})

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


def test_index_range_validation():
    with pytest.raises(ValueError, match="should be greater than or equal to 0"):
        IndexRange(start=-1, end=10)

    with pytest.raises(ValueError, match="should be greater than or equal to 0"):
        IndexRange(start=0, end=-1)

    with pytest.raises(ValueError, match="'start' index must be less than or equal to 'end' index"):
        IndexRange(start=11, end=10)


def test_index_range_size():
    assert IndexRange(start=0, end=10).size == 11
    assert IndexRange(start=1, end=10).size == 10
    assert IndexRange(start=0, end=0).size == 1


def test_partition_block_validation():
    with pytest.raises(ValueError, match="should be greater than or equal to 0"):
        PartitionBlock(index=-1, num_partitions=10)

    with pytest.raises(ValueError, match="should be greater than or equal to 1"):
        PartitionBlock(index=0, num_partitions=0)

    with pytest.raises(ValueError, match="'index' must be less than 'num_partitions'"):
        PartitionBlock(index=10, num_partitions=10)


def test_partition_block_to_index_range():
    index_range = PartitionBlock(index=0, num_partitions=10).to_index_range(101)
    assert index_range.start == 0
    assert index_range.end == 9
    assert index_range.size == 10

    index_range = PartitionBlock(index=1, num_partitions=10).to_index_range(105)
    assert index_range.start == 10
    assert index_range.end == 19
    assert index_range.size == 10

    index_range = PartitionBlock(index=2, num_partitions=10).to_index_range(105)
    assert index_range.start == 20
    assert index_range.end == 29
    assert index_range.size == 10

    index_range = PartitionBlock(index=9, num_partitions=10).to_index_range(105)
    assert index_range.start == 90
    assert index_range.end == 104
    assert index_range.size == 15


def test_local_seed_dataset_reference_validation(tmp_path: Path):
    with pytest.raises(InvalidFilePathError, match="🛑 Path test/dataset.parquet is not a file."):
        LocalSeedDatasetReference(dataset="test/dataset.parquet")

    # Should not raise an error when referencing supported extensions with wildcard pattern.
    create_partitions_in_path(tmp_path, "parquet")
    create_partitions_in_path(tmp_path, "csv")
    create_partitions_in_path(tmp_path, "json")
    create_partitions_in_path(tmp_path, "jsonl")

    test_cases = ["parquet", "csv", "json", "jsonl"]
    try:
        for extension in test_cases:
            reference = LocalSeedDatasetReference(dataset=f"{tmp_path}/*.{extension}")
            assert reference.dataset == f"{tmp_path}/*.{extension}"
    except Exception as e:
        pytest.fail(f"Expected no exception, but got {e}")


def test_local_seed_dataset_reference_validation_error(tmp_path: Path):
    create_partitions_in_path(tmp_path, "parquet")
    with pytest.raises(InvalidFilePathError, match="does not contain files of type 'csv'"):
        LocalSeedDatasetReference(dataset=f"{tmp_path}/*.csv")


def test_local_seed_dataset_reference_file_format_error(tmp_path: Path):
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    filepath = tmp_path / "test.txt"
    df.to_csv(filepath)

    with pytest.raises(InvalidFileFormatError):
        LocalSeedDatasetReference(dataset=filepath)


def _write_file(df: pd.DataFrame, path: Union[str, Path], file_type: str):
    if file_type == "parquet":
        df.to_parquet(path)
    elif file_type in {"json", "jsonl"}:
        df.to_json(path, orient="records", lines=True)
    else:
        df.to_csv(path, index=False)


@pytest.mark.parametrize("file_type", ["parquet", "json", "jsonl", "csv"])
def test_get_column_names(tmp_path, file_type):
    """Test get_file_column_names with basic parquet file."""
    test_data = {
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "city": ["NYC", "LA", "Chicago"],
    }
    df = pd.DataFrame(test_data)

    parquet_path = tmp_path / f"test_data.{file_type}"
    _write_file(df, parquet_path, file_type)

    reference = LocalSeedDatasetReference(dataset=parquet_path)
    assert reference.get_column_names() == df.columns.tolist()


def test_get_file_column_names_nested_fields(tmp_path):
    """Test get_file_column_names with nested fields in parquet."""
    schema = pa.schema(
        [
            pa.field(
                "nested", pa.struct([pa.field("col1", pa.list_(pa.int32())), pa.field("col2", pa.list_(pa.int32()))])
            ),
        ]
    )

    # For PyArrow, we need to structure the data as a list of records
    nested_data = {"nested": [{"col1": [1, 2, 3], "col2": [4, 5, 6]}]}
    nested_path = tmp_path / "nested_fields.parquet"
    pq.write_table(pa.Table.from_pydict(nested_data, schema=schema), nested_path)

    reference = LocalSeedDatasetReference(dataset=nested_path)
    column_names = reference.get_column_names()

    assert column_names == ["nested"]


@pytest.mark.parametrize("file_type", ["parquet", "json", "jsonl", "csv"])
def test_get_file_column_names_empty_parquet(tmp_path, file_type):
    """Test get_file_column_names with empty parquet file."""
    empty_df = pd.DataFrame()
    empty_path = tmp_path / f"empty.{file_type}"
    _write_file(empty_df, empty_path, file_type)

    reference = LocalSeedDatasetReference(dataset=empty_path)
    column_names = reference.get_column_names()

    assert column_names == []


@pytest.mark.parametrize("file_type", ["parquet", "json", "jsonl", "csv"])
def test_get_file_column_names_large_schema(tmp_path, file_type):
    """Test get_file_column_names with many columns."""
    num_columns = 50
    test_data = {f"col_{i}": np.random.randn(10) for i in range(num_columns)}
    df = pd.DataFrame(test_data)

    large_path = tmp_path / f"large_schema.{file_type}"
    _write_file(df, large_path, file_type)

    reference = LocalSeedDatasetReference(dataset=large_path)
    column_names = reference.get_column_names()

    assert len(column_names) == num_columns
    assert column_names == [f"col_{i}" for i in range(num_columns)]


@pytest.mark.parametrize("file_type", ["parquet", "json", "jsonl", "csv"])
def test_get_file_column_names_special_characters(tmp_path, file_type):
    """Test get_file_column_names with special characters in column names."""
    special_data = {
        "column with spaces": [1],
        "column-with-dashes": [2],
        "column_with_underscores": [3],
        "column.with.dots": [4],
        "column123": [5],
        "123column": [6],
        "column!@#$%^&*()": [7],
    }
    df_special = pd.DataFrame(special_data)
    special_path = tmp_path / f"special_chars.{file_type}"
    _write_file(df_special, special_path, file_type)

    reference = LocalSeedDatasetReference(dataset=special_path)
    column_names = reference.get_column_names()

    assert column_names == df_special.columns.tolist()


@pytest.mark.parametrize("file_type", ["parquet", "json", "jsonl", "csv"])
def test_get_file_column_names_unicode(tmp_path, file_type):
    """Test get_file_column_names with unicode column names."""
    unicode_data = {"café": [1], "résumé": [2], "naïve": [3], "façade": [4], "garçon": [5], "über": [6], "schön": [7]}
    df_unicode = pd.DataFrame(unicode_data)

    unicode_path = tmp_path / f"unicode_columns.{file_type}"
    _write_file(df_unicode, unicode_path, file_type)

    reference = LocalSeedDatasetReference(dataset=unicode_path)
    column_names = reference.get_column_names()

    assert column_names == df_unicode.columns.tolist()


@pytest.mark.parametrize("file_type", ["parquet", "csv", "json", "jsonl"])
def test_get_file_column_names_with_glob_pattern(tmp_path, file_type):
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    for i in range(5):
        _write_file(df, tmp_path / f"{i}.{file_type}", file_type)

    reference = LocalSeedDatasetReference(dataset=f"{tmp_path}/*.{file_type}")
    column_names = reference.get_column_names()

    assert column_names == ["col1", "col2"]


def test_get_file_column_names_error_handling(tmp_path):
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    filepath = tmp_path / "test.parquet"
    df.to_parquet(filepath)

    reference = LocalSeedDatasetReference(dataset=filepath)

    with patch("data_designer.config.seed.pq.read_schema") as mock_read_schema:
        mock_read_schema.side_effect = Exception("Test error")
        reference.get_column_names()

    with patch("data_designer.config.seed.pq.read_schema") as mock_read_schema:
        mock_col1 = MagicMock()
        mock_col1.name = "col1"
        mock_col2 = MagicMock()
        mock_col2.name = "col2"
        mock_read_schema.return_value = [mock_col1, mock_col2]

        column_names = reference.get_column_names()
        assert column_names == ["col1", "col2"]


TEST_ENDPOINT = "https://testing.com"
TEST_TOKEN = "stub-token"


def test_fetch_seed_dataset_column_names_parquet_error_handling():
    reference = HfHubSeedDatasetReference(
        dataset="test/repo/test.txt",
        endpoint=TEST_ENDPOINT,
        token=TEST_TOKEN,
    )
    with pytest.raises(InvalidFileFormatError, match="🛑 Unsupported file type: 'test.txt'"):
        reference.get_column_names()


@patch("data_designer.config.seed.HfFileSystem.open")
@patch("data_designer.config.seed._get_file_column_names", autospec=True)
def test_fetch_seed_dataset_column_names_remote_file(mock_get_file_column_names, mock_hf_fs_open):
    mock_get_file_column_names.return_value = ["col1", "col2"]

    reference = HfHubSeedDatasetReference(
        dataset="test/repo/test.parquet",
        endpoint=TEST_ENDPOINT,
        token=TEST_TOKEN,
    )

    assert reference.get_column_names() == ["col1", "col2"]
    mock_hf_fs_open.assert_called_once_with(
        "datasets/test/repo/test.parquet",
    )
