# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from data_designer.engine.resources.managed_assets import DatasetManager
from data_designer.engine.resources.managed_dataset_generator import ManagedDatasetGenerator
from data_designer.engine.resources.managed_dataset_repository import ManagedDatasetRepository
from data_designer.engine.sampling_gen.entities.person import load_person_data_sampler
from data_designer.engine.sampling_gen.errors import DatasetNotAvailableForLocaleError


@pytest.fixture
def stub_repository():
    mock_repo = Mock(spec=ManagedDatasetRepository)
    mock_repo.query.return_value = pd.DataFrame({"name": ["John", "Jane"], "age": [25, 30]})
    return mock_repo


@pytest.fixture
def stub_dataset_manager():
    return Mock(spec=DatasetManager)


@pytest.mark.parametrize(
    "dataset_name",
    ["en_US", "en_GB", "custom_dataset"],
)
def test_managed_dataset_generator_init(dataset_name, stub_repository):
    generator = ManagedDatasetGenerator(stub_repository, dataset_name=dataset_name)

    assert generator.dataset_repo == stub_repository
    assert generator.dataset_name == dataset_name


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
def test_generate_samples_scenarios(size, evidence, seed, expected_query_pattern, stub_repository):
    generator = ManagedDatasetGenerator(stub_repository, dataset_name="en_US")

    if size is None:
        result = generator.generate_samples(evidence=evidence, seed=seed)
    else:
        result = generator.generate_samples(size=size, evidence=evidence, seed=seed)

    stub_repository.query.assert_called_once()
    call_args = stub_repository.query.call_args[0][0]
    assert expected_query_pattern in call_args

    assert isinstance(result, pd.DataFrame)


def test_generate_samples_different_locale(stub_repository):
    generator = ManagedDatasetGenerator(stub_repository, dataset_name="ja_JP")

    result = generator.generate_samples(size=1)

    expected_query = "select * from 'ja_JP' order by random() limit 1"
    stub_repository.query.assert_called_once_with(expected_query)

    assert isinstance(result, pd.DataFrame)


@pytest.mark.parametrize(
    "locale",
    [
        "en_US",
        "ja_JP",
        "en_IN",
    ],
)
@patch("data_designer.engine.sampling_gen.entities.person.DuckDBDatasetRepository", autospec=True)
def test_load_person_data_sampler_scenarios(mock_repo_class, locale, stub_dataset_manager):
    mock_repo = Mock()
    mock_repo_class.return_value = mock_repo

    result = load_person_data_sampler(stub_dataset_manager, locale=locale)

    mock_repo_class.assert_called_once()
    call_kwargs = mock_repo_class.call_args[1]
    assert "use_cache" in call_kwargs

    assert isinstance(result, ManagedDatasetGenerator)
    assert result.dataset_repo == mock_repo
    assert result.dataset_name == locale


def test_load_person_data_sampler_invalid_locale(stub_dataset_manager):
    with pytest.raises(DatasetNotAvailableForLocaleError, match="Locale invalid_locale is not supported"):
        load_person_data_sampler(stub_dataset_manager, locale="invalid_locale")
