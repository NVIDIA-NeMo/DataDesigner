# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_designer.config.datastore import (
    DatastoreSettings,
    fetch_seed_dataset_column_names,
    get_file_column_names,
    resolve_datastore_settings,
    upload_to_hf_hub,
)
from data_designer.config.errors import InvalidConfigError, InvalidFileFormatError, InvalidFilePathError
from data_designer.config.seed import DatastoreSeedDatasetReference, LocalSeedDatasetReference


@pytest.fixture
def datastore_settings():
    return DatastoreSettings(endpoint="https://testing.com", token="stub-token")


def _write_file(df, path, file_type):
    if file_type == "parquet":
        df.to_parquet(path)
    elif file_type in {"json", "jsonl"}:
        df.to_json(path, orient="records", lines=True)
    else:
        df.to_csv(path, index=False)


@pytest.mark.parametrize("file_type", ["parquet", "json", "jsonl", "csv"])
def test_get_file_column_names_basic_parquet(tmp_path, file_type):
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
    assert get_file_column_names(str(parquet_path), file_type) == df.columns.tolist()


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

    column_names = get_file_column_names(str(nested_path), "parquet")

    assert column_names == ["nested"]


@pytest.mark.parametrize("file_type", ["parquet", "json", "jsonl", "csv"])
def test_get_file_column_names_empty_parquet(tmp_path, file_type):
    """Test get_file_column_names with empty parquet file."""
    empty_df = pd.DataFrame()
    empty_path = tmp_path / f"empty.{file_type}"
    _write_file(empty_df, empty_path, file_type)

    column_names = get_file_column_names(str(empty_path), file_type)
    assert column_names == []


@pytest.mark.parametrize("file_type", ["parquet", "json", "jsonl", "csv"])
def test_get_file_column_names_large_schema(tmp_path, file_type):
    """Test get_file_column_names with many columns."""
    num_columns = 50
    test_data = {f"col_{i}": np.random.randn(10) for i in range(num_columns)}
    df = pd.DataFrame(test_data)

    large_path = tmp_path / f"large_schema.{file_type}"
    _write_file(df, large_path, file_type)

    column_names = get_file_column_names(str(large_path), file_type)
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

    assert get_file_column_names(str(special_path), file_type) == df_special.columns.tolist()


@pytest.mark.parametrize("file_type", ["parquet", "json", "jsonl", "csv"])
def test_get_file_column_names_unicode(tmp_path, file_type):
    """Test get_file_column_names with unicode column names."""
    unicode_data = {"café": [1], "résumé": [2], "naïve": [3], "façade": [4], "garçon": [5], "über": [6], "schön": [7]}
    df_unicode = pd.DataFrame(unicode_data)

    unicode_path = tmp_path / f"unicode_columns.{file_type}"
    _write_file(df_unicode, unicode_path, file_type)
    assert get_file_column_names(str(unicode_path), file_type) == df_unicode.columns.tolist()


@pytest.mark.parametrize("file_type", ["parquet", "csv", "json", "jsonl"])
def test_get_file_column_names_with_glob_pattern(tmp_path, file_type):
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    for i in range(5):
        _write_file(df, tmp_path / f"{i}.{file_type}", file_type)
    assert get_file_column_names(f"{tmp_path}/*.{file_type}", file_type) == ["col1", "col2"]


def test_get_file_column_names_with_glob_pattern_error(tmp_path):
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    for i in range(5):
        _write_file(df, tmp_path / f"{i}.parquet", "parquet")
    with pytest.raises(InvalidFilePathError, match="No files found matching pattern"):
        get_file_column_names(f"{tmp_path}/*.csv", "csv")


def test_get_file_column_names_error_handling():
    with pytest.raises(InvalidFilePathError, match="🛑 Unsupported file type: 'txt'"):
        get_file_column_names("test.txt", "txt")

    with patch("data_designer.config.datastore.pq.read_schema") as mock_read_schema:
        mock_read_schema.side_effect = Exception("Test error")
        assert get_file_column_names("test.txt", "parquet") == []

    with patch("data_designer.config.datastore.pq.read_schema") as mock_read_schema:
        mock_col1 = MagicMock()
        mock_col1.name = "col1"
        mock_col2 = MagicMock()
        mock_col2.name = "col2"
        mock_read_schema.return_value = [mock_col1, mock_col2]
        assert get_file_column_names("test.txt", "parquet") == ["col1", "col2"]


def test_fetch_seed_dataset_column_names_parquet_error_handling(datastore_settings):
    with pytest.raises(InvalidFileFormatError, match="🛑 Unsupported file type: 'test.txt'"):
        fetch_seed_dataset_column_names(
            DatastoreSeedDatasetReference(
                dataset="test/repo/test.txt",
                datastore_settings=datastore_settings,
            )
        )


@patch("data_designer.config.datastore.get_file_column_names", autospec=True)
def test_fetch_seed_dataset_column_names_local_file(mock_get_file_column_names, datastore_settings):
    mock_get_file_column_names.return_value = ["col1", "col2"]
    with patch("data_designer.config.datastore.Path.is_file", autospec=True) as mock_is_file:
        mock_is_file.return_value = True
        assert fetch_seed_dataset_column_names(LocalSeedDatasetReference(dataset="test.parquet")) == ["col1", "col2"]


@patch("data_designer.config.datastore.HfFileSystem.open")
@patch("data_designer.config.datastore.get_file_column_names", autospec=True)
def test_fetch_seed_dataset_column_names_remote_file(mock_get_file_column_names, mock_hf_fs_open, datastore_settings):
    mock_get_file_column_names.return_value = ["col1", "col2"]
    assert fetch_seed_dataset_column_names(
        DatastoreSeedDatasetReference(
            dataset="test/repo/test.parquet",
            datastore_settings=datastore_settings,
        )
    ) == ["col1", "col2"]
    mock_hf_fs_open.assert_called_once_with(
        "datasets/test/repo/test.parquet",
    )


def test_resolve_datastore_settings(datastore_settings):
    with pytest.raises(InvalidConfigError, match="Datastore settings are required"):
        resolve_datastore_settings(None)

    with pytest.raises(InvalidConfigError, match="Invalid datastore settings format"):
        resolve_datastore_settings("invalid_settings")

    assert resolve_datastore_settings(datastore_settings) == datastore_settings
    assert resolve_datastore_settings(datastore_settings.model_dump()) == datastore_settings


@patch("data_designer.config.datastore.HfApi.upload_file", autospec=True)
@patch("data_designer.config.datastore.HfApi.create_repo", autospec=True)
def test_upload_to_hf_hub(mock_create_repo, mock_upload_file, datastore_settings):
    with patch("data_designer.config.datastore.Path.is_file", autospec=True) as mock_is_file:
        mock_is_file.return_value = True

        assert (
            upload_to_hf_hub("test.parquet", "test.parquet", "test/repo", datastore_settings)
            == "test/repo/test.parquet"
        )
        mock_create_repo.assert_called_once()
        mock_upload_file.assert_called_once()


def test_upload_to_hf_hub_error_handling(datastore_settings):
    with pytest.raises(
        InvalidFilePathError, match="To upload a dataset to the datastore, you must provide a valid file path."
    ):
        upload_to_hf_hub("test.txt", "test.txt", "test/repo", datastore_settings)

    with pytest.raises(
        InvalidFileFormatError, match="Dataset file extension '.parquet' does not match `filename` extension .'csv'"
    ):
        with patch("data_designer.config.datastore.Path.is_file", autospec=True) as mock_is_file:
            mock_is_file.return_value = True
            upload_to_hf_hub("test.parquet", "test.csv", "test/repo", datastore_settings)

    with pytest.raises(InvalidFileFormatError, match="Dataset files must be in "):
        with patch("data_designer.config.datastore.Path.is_file", autospec=True) as mock_is_file:
            mock_is_file.return_value = True
            upload_to_hf_hub("test.text", "test.txt", "test/repo", datastore_settings)
