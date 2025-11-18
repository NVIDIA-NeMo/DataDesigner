# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from faker import Faker
from faker.config import AVAILABLE_LOCALES as FAKER_AVAILABLE_LOCALES
import pytest

from data_designer.config.sampler_params import PersonSamplerParams, SamplerType
from data_designer.config.utils.constants import AVAILABLE_LOCALES
from data_designer.engine.sampling_gen.column import ConditionalDataColumn
from data_designer.engine.sampling_gen.errors import DatasetNotAvailableForLocaleError
from data_designer.engine.sampling_gen.people_gen import PeopleGenFaker, create_people_gen_resource
from data_designer.engine.sampling_gen.schema import DataSchema

NUM_PEOPLE = 100
FAKER_LOCALE = "en_GB"
PGM_LOCALE = "en_US"


def test_all_available_locales(stub_people_gen_resource):
    # Filter out deprecated locales to avoid warnings
    deprecated_locales = {"fr_QC"}  # fr_QC is deprecated, use fr_CA instead
    available_locales = [locale for locale in AVAILABLE_LOCALES if locale not in deprecated_locales]

    for locale in available_locales:
        if locale == PGM_LOCALE:
            pop = stub_people_gen_resource[locale].generate(NUM_PEOPLE)
        else:
            pop = PeopleGenFaker(Faker(locale), locale).generate(NUM_PEOPLE)
        assert {p["locale"] for p in pop} == {locale}
        assert len(pop) == NUM_PEOPLE


def test_available_locales_are_the_same_as_faker_available_locales():
    faker_locales = set(FAKER_AVAILABLE_LOCALES)
    faker_locales.remove("fr_QC")
    assert set(AVAILABLE_LOCALES) == set(faker_locales)


def test_people_fix_parameters_faker():
    pop = PeopleGenFaker(Faker(FAKER_LOCALE), FAKER_LOCALE).generate(NUM_PEOPLE, sex="Male", city="London")
    assert len(pop) == NUM_PEOPLE
    assert {p["sex"] for p in pop} == {"Male"}
    assert {p["city"] for p in pop} == {"London"}
    assert {p["locale"] for p in pop} == {FAKER_LOCALE}


def test_people_fix_parameters_pgm(stub_people_gen_pgm):
    pop = stub_people_gen_pgm.generate(NUM_PEOPLE, sex="Female", city="Brooklyn")
    assert len(pop) == NUM_PEOPLE
    assert {p["sex"] for p in pop} == {"Female"}
    assert {p["city"] for p in pop} == {"Brooklyn"}
    assert {p["locale"] for p in pop} == {PGM_LOCALE}


def test_people_with_personas_pgm(stub_people_gen_with_personas):
    pop = stub_people_gen_with_personas.generate(NUM_PEOPLE, with_synthetic_personas=True)
    assert len(pop) == NUM_PEOPLE
    assert {p["locale"] for p in pop} == {PGM_LOCALE}
    # Check that at least one person has the career_goals_and_ambitions field
    assert any("career_goals_and_ambitions" in p for p in pop)


def test_create_people_gen_resource_with_valid_locale():
    """Test that create_people_gen_resource succeeds with a valid locale."""
    mock_dataset_repository = Mock()
    mock_dataset_repository.has_access_to_table.return_value = True
    mock_table = Mock(name="en_us")
    mock_dataset_repository.get_table.return_value = mock_table
    mock_dataset_repository.get_all_tables.return_value = [mock_table]

    params = PersonSamplerParams(locale="en_US", age_range=[18, 65])
    column = ConditionalDataColumn(
        name="person",
        sampler_type=SamplerType.PERSON,
        params=params,
    )
    schema = DataSchema(columns=[column])

    result = create_people_gen_resource(schema, mock_dataset_repository)

    mock_dataset_repository.has_access_to_table.assert_called_once_with("en_us")
    assert params.people_gen_key in result


def test_create_people_gen_resource_raises_error_for_unavailable_locale():
    """Test that create_people_gen_resource raises DatasetNotAvailableForLocaleError for unavailable locale."""
    mock_dataset_repository = Mock()
    mock_dataset_repository.has_access_to_table.return_value = False
    mock_dataset_repository.get_all_tables.return_value = []

    params = PersonSamplerParams(locale="en_US", age_range=[18, 65])
    column = ConditionalDataColumn(
        name="person",
        sampler_type=SamplerType.PERSON,
        params=params,
    )
    schema = DataSchema(columns=[column])

    with pytest.raises(DatasetNotAvailableForLocaleError) as exc_info:
        create_people_gen_resource(schema, mock_dataset_repository)

    assert "en_US" in str(exc_info.value)
    assert "not available" in str(exc_info.value).lower()
    mock_dataset_repository.has_access_to_table.assert_called_once_with("en_us")


def test_create_people_gen_resource_validates_all_conditional_params():
    """Test that create_people_gen_resource validates locales in conditional params."""
    mock_dataset_repository = Mock()

    def has_access_side_effect(table_name: str, exact_match: bool = False) -> bool:
        return table_name == "en_us"

    mock_dataset_repository.has_access_to_table.side_effect = has_access_side_effect

    base_params = PersonSamplerParams(locale="en_US", age_range=[18, 65])
    conditional_params = {"condition": PersonSamplerParams(locale="ja_JP", age_range=[25, 45])}

    column = ConditionalDataColumn(
        name="person",
        sampler_type=SamplerType.PERSON,
        params=base_params,
        conditional_params=conditional_params,
    )
    schema = DataSchema(columns=[column])

    with pytest.raises(DatasetNotAvailableForLocaleError) as exc_info:
        create_people_gen_resource(schema, mock_dataset_repository)

    assert "ja_JP" in str(exc_info.value)
    assert "not available" in str(exc_info.value).lower()


def test_create_people_gen_resource_multiple_columns():
    """Test that create_people_gen_resource handles multiple person columns correctly."""
    mock_dataset_repository = Mock()
    mock_dataset_repository.has_access_to_table.return_value = True

    mock_table_us = Mock(name="en_us")
    mock_table_jp = Mock(name="ja_jp")

    def get_table_side_effect(table_name: str):
        if table_name == "en_us":
            return mock_table_us
        elif table_name == "ja_jp":
            return mock_table_jp
        return Mock(name=table_name)

    mock_dataset_repository.get_table.side_effect = get_table_side_effect
    mock_dataset_repository.get_all_tables.return_value = [mock_table_us, mock_table_jp]

    params1 = PersonSamplerParams(locale="en_US", age_range=[18, 65])
    params2 = PersonSamplerParams(locale="ja_JP", age_range=[25, 45])

    column1 = ConditionalDataColumn(
        name="person_us",
        sampler_type=SamplerType.PERSON,
        params=params1,
    )
    column2 = ConditionalDataColumn(
        name="person_jp",
        sampler_type=SamplerType.PERSON,
        params=params2,
    )
    schema = DataSchema(columns=[column1, column2])

    result = create_people_gen_resource(schema, mock_dataset_repository)

    assert mock_dataset_repository.has_access_to_table.call_count == 2
    assert params1.people_gen_key in result
    assert params2.people_gen_key in result


def test_create_people_gen_resource_with_personas():
    """Test that create_people_gen_resource correctly handles with_synthetic_personas parameter."""
    mock_dataset_repository = Mock()
    mock_dataset_repository.has_access_to_table.return_value = True
    mock_dataset_repository.get_table.return_value = Mock(name="en_us_v1")

    params = PersonSamplerParams(locale="en_US", age_range=[18, 65], with_synthetic_personas=True)
    column = ConditionalDataColumn(
        name="person",
        sampler_type=SamplerType.PERSON,
        params=params,
    )
    schema = DataSchema(columns=[column])

    result = create_people_gen_resource(schema, mock_dataset_repository)

    assert params.people_gen_key == "en_US_with_personas"
    assert params.people_gen_key in result
