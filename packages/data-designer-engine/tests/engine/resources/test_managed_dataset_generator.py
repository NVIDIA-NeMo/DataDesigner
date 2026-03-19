# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.resources.managed_dataset_generator import ManagedDatasetGenerator
from data_designer.engine.resources.person_reader import PersonReader
from data_designer.engine.sampling_gen.entities.person import load_person_data_sampler
from data_designer.engine.sampling_gen.errors import DatasetNotAvailableForLocaleError


@pytest.fixture
def stub_reader() -> Mock:
    mock_reader = Mock(spec=PersonReader)
    mock_reader.get_dataset_uri.return_value = "/data/datasets/en_US.parquet"
    mock_reader.execute.return_value = lazy.pd.DataFrame({"name": ["John", "Jane"], "age": [25, 30]})
    return mock_reader


@pytest.mark.parametrize(
    "locale",
    ["en_US", "en_GB", "custom_dataset"],
)
def test_managed_dataset_generator_init(locale: str, stub_reader: Mock) -> None:
    generator = ManagedDatasetGenerator(stub_reader, locale=locale)

    assert generator._person_reader is stub_reader
    assert generator._locale == locale


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
) -> None:
    generator = ManagedDatasetGenerator(stub_reader, locale="en_US")

    if size is None:
        result = generator.generate_samples(evidence=evidence)
    else:
        result = generator.generate_samples(size=size, evidence=evidence)

    stub_reader.execute.assert_called_once_with(expected_query_pattern, expected_parameters)

    assert isinstance(result, lazy.pd.DataFrame)


def test_generate_samples_different_locale(stub_reader: Mock) -> None:
    stub_reader.get_dataset_uri.return_value = "/data/datasets/ja_JP.parquet"
    generator = ManagedDatasetGenerator(stub_reader, locale="ja_JP")

    result = generator.generate_samples(size=1)

    stub_reader.get_dataset_uri.assert_called_once_with("ja_JP")
    stub_reader.execute.assert_called_once()
    call_args = stub_reader.execute.call_args[0][0]
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
    mock_reader = Mock(spec=PersonReader)
    mock_reader.get_dataset_uri.return_value = f"/data/datasets/{locale}.parquet"

    result = load_person_data_sampler(mock_reader, locale=locale)

    assert isinstance(result, ManagedDatasetGenerator)


def test_load_person_data_sampler_invalid_locale() -> None:
    mock_reader = Mock(spec=PersonReader)
    with pytest.raises(DatasetNotAvailableForLocaleError, match="Locale invalid_locale is not supported"):
        load_person_data_sampler(mock_reader, locale="invalid_locale")
