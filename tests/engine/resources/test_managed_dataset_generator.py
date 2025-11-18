# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from data_designer.engine.resources.managed_dataset_generator import DuckDBDatasetGenerator
from data_designer.engine.resources.sampler_dataset_repository import SamplerDatasetRepository


@pytest.fixture
def stub_dataset_repository():
    return Mock(spec=SamplerDatasetRepository)


@patch("data_designer.engine.resources.managed_dataset_generator.duckdb", autospec=True)
def test_managed_dataset_generator_init(mock_duckdb, stub_dataset_repository):
    mock_db = Mock()
    mock_duckdb.connect.return_value = mock_db

    with patch("data_designer.engine.resources.managed_dataset_generator.threading.Thread"):
        generator = DuckDBDatasetGenerator(stub_dataset_repository)

        assert generator._dataset_repository == stub_dataset_repository
        assert generator._config == {"threads": 1, "memory_limit": "2 gb"}
        assert generator._use_cache is True
        assert generator.db == mock_db


@pytest.mark.parametrize(
    "size,evidence,seed,expected_query_pattern",
    [
        (2, None, None, "select * from 'en_US' order by random() limit 2"),
        (
            1,
            {"name": "John"},
            None,
            "select * from 'en_US' where name IN ('John') order by random() limit 1",
        ),
        (
            3,
            {"name": ["John", "Jane"], "age": [25]},
            None,
            "select * from 'en_US' where name IN ('John', 'Jane') and age IN ('25') order by random() limit 3",
        ),
        (
            1,
            {"name": [], "age": None},
            None,
            "select * from 'en_US' order by random() limit 1",
        ),
        (1, None, 12345, "select * from 'en_US' order by random() limit 1"),
        (
            None,
            None,
            None,
            "select * from 'en_US' order by random() limit 1",
        ),
    ],
)
@patch("data_designer.engine.resources.managed_dataset_generator.duckdb", autospec=True)
def test_generate_samples_scenarios(mock_duckdb, stub_dataset_repository, size, evidence, seed, expected_query_pattern):
    mock_db = Mock()
    mock_cursor = Mock()
    mock_df = pd.DataFrame({"col1": [1, 2, 3]})

    mock_duckdb.connect.return_value = mock_db
    mock_db.cursor.return_value = mock_cursor
    mock_cursor.sql.return_value.df.return_value = mock_df

    with patch("data_designer.engine.resources.managed_dataset_generator.threading.Thread"):
        generator = DuckDBDatasetGenerator(stub_dataset_repository)
        generator._registration_event.set()

        if size is None:
            result = generator.generate_samples_from_table(table_name="en_US", evidence=evidence, seed=seed)
        else:
            result = generator.generate_samples_from_table(table_name="en_US", size=size, evidence=evidence, seed=seed)

        mock_cursor.sql.assert_called_once()
        call_args = mock_cursor.sql.call_args[0][0]
        assert expected_query_pattern in call_args

        assert isinstance(result, pd.DataFrame)


@patch("data_designer.engine.resources.managed_dataset_generator.duckdb", autospec=True)
def test_generate_samples_different_locale(mock_duckdb, stub_dataset_repository):
    mock_db = Mock()
    mock_cursor = Mock()
    mock_df = pd.DataFrame({"col1": [1]})

    mock_duckdb.connect.return_value = mock_db
    mock_db.cursor.return_value = mock_cursor
    mock_cursor.sql.return_value.df.return_value = mock_df

    with patch("data_designer.engine.resources.managed_dataset_generator.threading.Thread"):
        generator = DuckDBDatasetGenerator(stub_dataset_repository)
        generator._registration_event.set()

        result = generator.generate_samples_from_table(table_name="ja_JP", size=1)

        expected_query = "select * from 'ja_JP' order by random() limit 1"
        mock_cursor.sql.assert_called_once_with(expected_query)

        assert isinstance(result, pd.DataFrame)
