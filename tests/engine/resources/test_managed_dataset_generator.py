# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from data_designer.engine.resources.managed_dataset_generator import ManagedDatasetGenerator
from data_designer.engine.resources.managed_dataset_repository import ManagedDatasetRepository
from data_designer.engine.resources.managed_storage import ManagedBlobStorage
from data_designer.engine.sampling_gen.entities.person import load_person_data_sampler
from data_designer.engine.sampling_gen.errors import DatasetNotAvailableForLocaleError


@pytest.fixture
def stub_repository():
    mock_repo = Mock(spec=ManagedDatasetRepository)
    mock_repo.query.return_value = pd.DataFrame({"name": ["John", "Jane"], "age": [25, 30]})
    return mock_repo


@pytest.fixture
def stub_blob_storage():
    return Mock(spec=ManagedBlobStorage)


@pytest.mark.parametrize(
    "dataset_name",
    ["en_US", "en_GB", "custom_dataset"],
)
def test_managed_dataset_generator_init(dataset_name, stub_repository):
    generator = ManagedDatasetGenerator(stub_repository, dataset_name=dataset_name)

    assert generator.managed_datasets == stub_repository
    assert generator.dataset_name == dataset_name


@pytest.mark.parametrize(
    "size,evidence,expected_query_pattern,expected_parameters",
    [
        (2, None, "select * from en_US order by random() limit 2", []),
        (
            1,
            {"name": "John"},
            "select * from en_US where name IN (?) order by random() limit 1",
            ["John"],
        ),
        (
            3,
            {"name": ["John", "Jane"], "age": [25]},
            "select * from en_US where name IN (?, ?) and age IN (?) order by random() limit 3",
            ["John", "Jane", 25],
        ),
        (
            1,
            {"name": [], "age": None},
            "select * from en_US order by random() limit 1",
            [],
        ),
        (
            None,
            None,
            "select * from en_US order by random() limit 1",
            [],
        ),
    ],
)
def test_generate_samples_scenarios(size, evidence, expected_query_pattern, expected_parameters, stub_repository):
    generator = ManagedDatasetGenerator(stub_repository, dataset_name="en_US")

    if size is None:
        result = generator.generate_samples(evidence=evidence)
    else:
        result = generator.generate_samples(size=size, evidence=evidence)

    stub_repository.query.assert_called_once_with(expected_query_pattern, expected_parameters)

    assert isinstance(result, pd.DataFrame)


def test_generate_samples_different_locale(stub_repository):
    generator = ManagedDatasetGenerator(stub_repository, dataset_name="ja_JP")

    result = generator.generate_samples(size=1)

    stub_repository.query.assert_called_once()
    call_args = stub_repository.query.call_args[0][0]
    expected_query = "select * from ja_JP order by random() limit 1"
    assert expected_query in call_args

    assert isinstance(result, pd.DataFrame)


@pytest.mark.parametrize(
    "locale",
    [
        "en_US",
        "ja_JP",
        "en_IN",
    ],
)
@patch("data_designer.engine.sampling_gen.entities.person.load_managed_dataset_repository", autospec=True)
def test_load_person_data_sampler_scenarios(mock_load_repo, locale, stub_blob_storage):
    mock_repo = Mock()
    mock_load_repo.return_value = mock_repo

    result = load_person_data_sampler(stub_blob_storage, locale=locale)

    mock_load_repo.assert_called_once_with(stub_blob_storage, [locale])

    assert isinstance(result, ManagedDatasetGenerator)
    assert result.managed_datasets == mock_repo
    assert result.dataset_name == locale


def test_load_person_data_sampler_invalid_locale(stub_blob_storage):
    with pytest.raises(DatasetNotAvailableForLocaleError, match="Locale invalid_locale is not supported"):
        load_person_data_sampler(stub_blob_storage, locale="invalid_locale")
