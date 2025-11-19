# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from data_designer.engine.resources.managed_dataset_repository import (
    DATASETS_ROOT,
    DEFAULT_DATA_CATALOG,
    DuckDBDatasetRepository,
    Table,
    load_managed_dataset_repository,
)
from data_designer.engine.resources.managed_storage import ManagedBlobStorage


def test_table_creation_default_schema():
    table = Table("test_file.parquet")

    assert table.source == "test_file.parquet"
    assert table.schema == "main"
    assert table.name == "test_file"


def test_table_creation_custom_schema():
    table = Table("test_file.parquet", schema="custom")

    assert table.source == "test_file.parquet"
    assert table.schema == "custom"
    assert table.name == "test_file"


def test_table_name_property():
    table = Table("path/to/test_file.parquet")
    assert table.name == "test_file"

    table2 = Table("another_file.csv")
    assert table2.name == "another_file"


@pytest.fixture
def stub_blob_storage():
    mock_storage = Mock(spec=ManagedBlobStorage)
    mock_storage.uri_for_key.return_value = "file://test/path"
    return mock_storage


@pytest.fixture
def stub_test_data_catalog():
    return [
        Table("test1.parquet"),
        Table("test2.parquet", schema="custom"),
    ]


@patch("data_designer.engine.resources.managed_dataset_repository.duckdb", autospec=True)
def test_duckdb_dataset_repository_init_default_config(mock_duckdb, stub_blob_storage):
    mock_db = Mock()
    mock_duckdb.connect.return_value = mock_db

    with patch("data_designer.engine.resources.managed_dataset_repository.threading.Thread") as mock_thread:
        repo = DuckDBDatasetRepository(stub_blob_storage)

        mock_duckdb.connect.assert_called_once_with(config={"threads": 2, "memory_limit": "4 gb"})

        assert repo._data_catalog == DEFAULT_DATA_CATALOG
        assert repo._data_sets_root == DATASETS_ROOT
        assert repo._blob_storage == stub_blob_storage
        assert repo._config == {"threads": 2, "memory_limit": "4 gb"}
        assert repo._use_cache is True
        assert repo.db == mock_db

        mock_thread.assert_called_once()


@patch("data_designer.engine.resources.managed_dataset_repository.duckdb", autospec=True)
def test_duckdb_dataset_repository_init_custom_config(mock_duckdb, stub_blob_storage, stub_test_data_catalog):
    mock_db = Mock()
    mock_duckdb.connect.return_value = mock_db

    custom_config = {"threads": 4, "memory_limit": "8 gb"}

    with patch("data_designer.engine.resources.managed_dataset_repository.threading.Thread"):
        repo = DuckDBDatasetRepository(
            stub_blob_storage,
            config=custom_config,
            data_catalog=stub_test_data_catalog,
            datasets_root="custom_root",
            use_cache=False,
        )

        mock_duckdb.connect.assert_called_once_with(config=custom_config)

        assert repo._data_catalog == stub_test_data_catalog
        assert repo._data_sets_root == "custom_root"
        assert repo._config == custom_config
        assert repo._use_cache is False


@patch("data_designer.engine.resources.managed_dataset_repository.duckdb", autospec=True)
def test_duckdb_dataset_repository_data_catalog_property(mock_duckdb, stub_blob_storage, stub_test_data_catalog):
    mock_db = Mock()
    mock_duckdb.connect.return_value = mock_db

    with patch("data_designer.engine.resources.managed_dataset_repository.threading.Thread"):
        repo = DuckDBDatasetRepository(stub_blob_storage, data_catalog=stub_test_data_catalog)

        assert repo.data_catalog == stub_test_data_catalog


@patch("data_designer.engine.resources.managed_dataset_repository.duckdb", autospec=True)
def test_duckdb_dataset_repository_query_basic(mock_duckdb, stub_blob_storage):
    mock_db = Mock()
    mock_cursor = Mock()
    mock_df = pd.DataFrame({"col1": [1, 2, 3]})

    mock_duckdb.connect.return_value = mock_db
    mock_db.cursor.return_value = mock_cursor
    mock_cursor.execute.return_value.df.return_value = mock_df

    with patch("data_designer.engine.resources.managed_dataset_repository.threading.Thread"):
        repo = DuckDBDatasetRepository(stub_blob_storage)

        repo._registration_event.set()

        result = repo.query("SELECT * FROM test", [])

        mock_db.cursor.assert_called_once()
        mock_cursor.execute.assert_called_once_with("SELECT * FROM test", [])
        mock_cursor.close.assert_called_once()

        pd.testing.assert_frame_equal(result, mock_df)


@patch("data_designer.engine.resources.managed_dataset_repository.duckdb", autospec=True)
def test_duckdb_dataset_repository_query_waits_for_registration(mock_duckdb, stub_blob_storage):
    mock_db = Mock()
    mock_cursor = Mock()
    mock_df = pd.DataFrame({"col1": [1, 2, 3]})

    mock_duckdb.connect.return_value = mock_db
    mock_db.cursor.return_value = mock_cursor
    mock_cursor.execute.return_value.df.return_value = mock_df

    with patch("data_designer.engine.resources.managed_dataset_repository.threading.Thread"):
        repo = DuckDBDatasetRepository(stub_blob_storage)

        repo._registration_event.clear()

        def mock_wait():
            repo._registration_event.set()

        repo._registration_event.wait = mock_wait

        result = repo.query("SELECT * FROM test", [])

        mock_cursor.execute.assert_called_once_with("SELECT * FROM test", [])
        pd.testing.assert_frame_equal(result, mock_df)


@patch("data_designer.engine.resources.managed_dataset_repository.duckdb", autospec=True)
def test_duckdb_dataset_repository_query_cursor_cleanup(mock_duckdb, stub_blob_storage):
    mock_db = Mock()
    mock_cursor = Mock()

    mock_duckdb.connect.return_value = mock_db
    mock_db.cursor.return_value = mock_cursor
    mock_cursor.execute.side_effect = Exception("Query failed")

    with patch("data_designer.engine.resources.managed_dataset_repository.threading.Thread"):
        repo = DuckDBDatasetRepository(stub_blob_storage)
        repo._registration_event.set()

        with pytest.raises(Exception, match="Query failed"):
            repo.query("SELECT * FROM test", [])

        mock_cursor.close.assert_called_once()


def test_load_managed_dataset_repository_with_local_storage(stub_blob_storage):
    with patch("data_designer.engine.resources.managed_dataset_repository.DuckDBDatasetRepository") as mock_repo_class:
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo

        locales = ["en_US", "ja_JP"]
        result = load_managed_dataset_repository(stub_blob_storage, locales)

        mock_repo_class.assert_called_once_with(
            stub_blob_storage,
            config={"threads": 1, "memory_limit": "2 gb"},
            data_catalog=[Table("en_US.parquet"), Table("ja_JP.parquet")],
            use_cache=True,  # Mock is not LocalBlobStorageProvider, so use_cache=True
        )

        assert result == mock_repo


def test_load_managed_dataset_repository_with_non_local_storage(stub_blob_storage):
    with patch("data_designer.engine.resources.managed_dataset_repository.DuckDBDatasetRepository") as mock_repo_class:
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo

        locales = ["en_US"]
        result = load_managed_dataset_repository(stub_blob_storage, locales)

        mock_repo_class.assert_called_once_with(
            stub_blob_storage,
            config={"threads": 1, "memory_limit": "2 gb"},
            data_catalog=[Table("en_US.parquet")],
            use_cache=True,  # Should be True for non-local storage
        )

        assert result == mock_repo
