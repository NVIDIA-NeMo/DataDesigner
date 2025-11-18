# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from data_designer.engine.resources.catalogs import Table
from data_designer.engine.resources.managed_dataset_generator import DuckDBDatasetGenerator
from data_designer.engine.resources.sampler_dataset_repository import SamplerDatasetRepository


@pytest.fixture
def stub_dataset_repository():
    mock_manager = Mock(spec=SamplerDatasetRepository)
    mock_table1 = Table("test1.parquet", "test1")
    mock_table2 = Table("test2.parquet", "test2")
    mock_manager.get_all_tables.return_value = [mock_table1, mock_table2]
    return mock_manager


@pytest.fixture
def stub_test_data_catalog():
    return [
        Table("test1.parquet", "test1"),
        Table("test2.parquet", "test2"),
    ]


@patch("data_designer.engine.resources.managed_dataset_generator.duckdb", autospec=True)
def test_duckdb_dataset_repository_init_default_config(mock_duckdb, stub_dataset_repository):
    mock_db = Mock()
    mock_duckdb.connect.return_value = mock_db

    with patch("data_designer.engine.resources.managed_dataset_generator.threading.Thread") as mock_thread:
        repo = DuckDBDatasetGenerator(stub_dataset_repository)

        mock_duckdb.connect.assert_called_once_with(config={"threads": 1, "memory_limit": "2 gb"})

        assert repo._dataset_repository == stub_dataset_repository
        assert repo._config == {"threads": 1, "memory_limit": "2 gb"}
        assert repo._use_cache is True
        assert repo.db == mock_db

        mock_thread.assert_called_once()


@patch("data_designer.engine.resources.managed_dataset_generator.duckdb", autospec=True)
def test_duckdb_dataset_repository_init_custom_config(mock_duckdb, stub_dataset_repository):
    mock_db = Mock()
    mock_duckdb.connect.return_value = mock_db

    custom_config = {"threads": 4, "memory_limit": "8 gb"}

    with patch("data_designer.engine.resources.managed_dataset_generator.threading.Thread"):
        repo = DuckDBDatasetGenerator(
            stub_dataset_repository,
            config=custom_config,
            use_cache=False,
        )

        mock_duckdb.connect.assert_called_once_with(config=custom_config)

        assert repo._dataset_repository == stub_dataset_repository
        assert repo._config == custom_config
        assert repo._use_cache is False


@patch("data_designer.engine.resources.managed_dataset_generator.duckdb", autospec=True)
def test_duckdb_dataset_repository_data_catalog_property(mock_duckdb, stub_dataset_repository, stub_test_data_catalog):
    mock_db = Mock()
    mock_duckdb.connect.return_value = mock_db
    stub_dataset_repository.get_data_catalog.return_value = stub_test_data_catalog

    with patch("data_designer.engine.resources.managed_dataset_generator.threading.Thread"):
        repo = DuckDBDatasetGenerator(stub_dataset_repository)

        assert repo._dataset_repository.get_data_catalog("test_catalog") == stub_test_data_catalog


@patch("data_designer.engine.resources.managed_dataset_generator.duckdb", autospec=True)
def test_duckdb_dataset_repository_query_basic(mock_duckdb, stub_dataset_repository):
    mock_db = Mock()
    mock_cursor = Mock()
    mock_df = pd.DataFrame({"col1": [1, 2, 3]})

    mock_duckdb.connect.return_value = mock_db
    mock_db.cursor.return_value = mock_cursor
    mock_cursor.sql.return_value.df.return_value = mock_df

    with patch("data_designer.engine.resources.managed_dataset_generator.threading.Thread"):
        repo = DuckDBDatasetGenerator(stub_dataset_repository)

        repo._registration_event.set()

        result = repo.query("SELECT * FROM test")

        mock_db.cursor.assert_called_once()
        mock_cursor.sql.assert_called_once_with("SELECT * FROM test")
        mock_cursor.close.assert_called_once()

        pd.testing.assert_frame_equal(result, mock_df)


@patch("data_designer.engine.resources.managed_dataset_generator.duckdb", autospec=True)
def test_duckdb_dataset_repository_query_waits_for_registration(mock_duckdb, stub_dataset_repository):
    mock_db = Mock()
    mock_cursor = Mock()
    mock_df = pd.DataFrame({"col1": [1, 2, 3]})

    mock_duckdb.connect.return_value = mock_db
    mock_db.cursor.return_value = mock_cursor
    mock_cursor.sql.return_value.df.return_value = mock_df

    with patch("data_designer.engine.resources.managed_dataset_generator.threading.Thread"):
        repo = DuckDBDatasetGenerator(stub_dataset_repository)

        repo._registration_event.clear()

        def mock_wait():
            repo._registration_event.set()

        repo._registration_event.wait = mock_wait

        result = repo.query("SELECT * FROM test")

        mock_cursor.sql.assert_called_once_with("SELECT * FROM test")
        pd.testing.assert_frame_equal(result, mock_df)


@patch("data_designer.engine.resources.managed_dataset_generator.duckdb", autospec=True)
def test_duckdb_dataset_repository_query_cursor_cleanup(mock_duckdb, stub_dataset_repository):
    mock_db = Mock()
    mock_cursor = Mock()

    mock_duckdb.connect.return_value = mock_db
    mock_db.cursor.return_value = mock_cursor
    mock_cursor.sql.side_effect = Exception("Query failed")

    with patch("data_designer.engine.resources.managed_dataset_generator.threading.Thread"):
        repo = DuckDBDatasetGenerator(stub_dataset_repository)
        repo._registration_event.set()

        with pytest.raises(Exception, match="Query failed"):
            repo.query("SELECT * FROM test")

        mock_cursor.close.assert_called_once()
