# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.resources.managed_dataset_generator import ManagedDatasetGenerator
from data_designer.engine.resources.nemotron_personas_reader import NemotronPersonasDatasetReader
from data_designer.engine.sampling_gen.entities.person import load_person_data_sampler
from data_designer.engine.sampling_gen.errors import DatasetNotAvailableForLocaleError


@pytest.fixture
def stub_reader() -> Mock:
    mock_reader = Mock(spec=NemotronPersonasDatasetReader)
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_cursor.execute.return_value.df.return_value = lazy.pd.DataFrame({"name": ["John", "Jane"], "age": [25, 30]})
    mock_conn.cursor.return_value = mock_cursor
    mock_reader.create_duckdb_connection.return_value = mock_conn
    mock_reader.get_dataset_uri.return_value = "/data/datasets/en_US.parquet"
    return mock_reader


@pytest.fixture
def stub_cursor(stub_reader: Mock) -> Mock:
    return stub_reader.create_duckdb_connection.return_value.cursor.return_value


@pytest.mark.parametrize(
    "locale",
    ["en_US", "en_GB", "custom_dataset"],
)
def test_managed_dataset_generator_init(locale: str, stub_reader: Mock) -> None:
    stub_reader.get_dataset_uri.return_value = f"/data/datasets/{locale}.parquet"
    generator = ManagedDatasetGenerator(stub_reader, locale=locale)

    stub_reader.create_duckdb_connection.assert_called_once()
    stub_reader.get_dataset_uri.assert_called_once_with(locale)
    assert generator._uri == f"/data/datasets/{locale}.parquet"


@pytest.mark.parametrize(
    "size,evidence,expected_query_pattern,expected_parameters",
    [
        (2, None, "select * from '/data/datasets/en_US.parquet' order by random() limit 2", []),
        (
            1,
            {"name": "John"},
            "select * from '/data/datasets/en_US.parquet' where name IN (?) order by random() limit 1",
            ["John"],
        ),
        (
            3,
            {"name": ["John", "Jane"], "age": [25]},
            "select * from '/data/datasets/en_US.parquet' where name IN (?, ?) and age IN (?) order by random() limit 3",
            ["John", "Jane", 25],
        ),
        (
            1,
            {"name": [], "age": None},
            "select * from '/data/datasets/en_US.parquet' order by random() limit 1",
            [],
        ),
        (
            None,
            None,
            "select * from '/data/datasets/en_US.parquet' order by random() limit 1",
            [],
        ),
    ],
)
def test_generate_samples_scenarios(
    size: int | None,
    evidence: dict | None,
    expected_query_pattern: str,
    expected_parameters: list,
    stub_reader: Mock,
    stub_cursor: Mock,
) -> None:
    generator = ManagedDatasetGenerator(stub_reader, locale="en_US")

    if size is None:
        result = generator.generate_samples(evidence=evidence)
    else:
        result = generator.generate_samples(size=size, evidence=evidence)

    stub_cursor.execute.assert_called_once_with(expected_query_pattern, expected_parameters)
    stub_cursor.close.assert_called_once()

    assert isinstance(result, lazy.pd.DataFrame)


def test_generate_samples_different_locale(stub_reader: Mock, stub_cursor: Mock) -> None:
    stub_reader.get_dataset_uri.return_value = "/data/datasets/ja_JP.parquet"
    generator = ManagedDatasetGenerator(stub_reader, locale="ja_JP")

    result = generator.generate_samples(size=1)

    stub_cursor.execute.assert_called_once()
    call_args = stub_cursor.execute.call_args[0][0]
    assert "'/data/datasets/ja_JP.parquet'" in call_args

    assert isinstance(result, lazy.pd.DataFrame)


@pytest.mark.parametrize(
    "locale",
    [
        "en_US",
        "ja_JP",
        "en_IN",
    ],
)
def test_load_person_data_sampler_scenarios(locale: str) -> None:
    mock_reader = Mock(spec=NemotronPersonasDatasetReader)
    mock_conn = Mock()
    mock_reader.create_duckdb_connection.return_value = mock_conn
    mock_reader.get_dataset_uri.return_value = f"/data/datasets/{locale}.parquet"

    result = load_person_data_sampler(mock_reader, locale=locale)

    mock_reader.create_duckdb_connection.assert_called_once()
    mock_reader.get_dataset_uri.assert_called_once_with(locale)
    assert isinstance(result, ManagedDatasetGenerator)


def test_load_person_data_sampler_invalid_locale() -> None:
    mock_reader = Mock(spec=NemotronPersonasDatasetReader)
    with pytest.raises(DatasetNotAvailableForLocaleError, match="Locale invalid_locale is not supported"):
        load_person_data_sampler(mock_reader, locale="invalid_locale")
