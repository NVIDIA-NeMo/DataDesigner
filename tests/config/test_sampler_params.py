# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import ValidationError
import pytest

from data_designer.config.sampler_params import (
    CategorySamplerParams,
    DatetimeSamplerParams,
    PersonSamplerParams,
    SamplerType,
    TimeDeltaSamplerParams,
    UUIDSamplerParams,
    is_numerical_sampler_type,
)


@pytest.fixture
def stub_person_sampler_params():
    return PersonSamplerParams(locale="en_US", sex="Female", city="New York", age_range=[18, 30])


def test_category_sampler_params():
    params = CategorySamplerParams(values=["a", "b", "c"], weights=[2, 5, 3])
    assert params.values == ["a", "b", "c"]
    assert params.weights == [0.2, 0.5, 0.3]

    with pytest.raises(ValueError, match="'categories' and 'weights' must have the same length"):
        CategorySamplerParams(values=["a", "b", "c"], weights=[10, 1])


def test_datetime_sampler_params():
    params = DatetimeSamplerParams(start="2020-01-01", end="2025-01-01", unit="D")
    assert params.start == "2020-01-01"
    assert params.end == "2025-01-01"
    assert params.unit == "D"

    with pytest.raises(ValueError, match="Invalid datetime format"):
        DatetimeSamplerParams(start="invalid", end="invalid", unit="D")


def test_timedelta_sampler_params():
    params = TimeDeltaSamplerParams(dt_min=1, dt_max=10, unit="D", reference_column_name="datetime")
    assert params.dt_min == 1
    assert params.dt_max == 10
    assert params.unit == "D"
    assert params.reference_column_name == "datetime"

    with pytest.raises(ValueError, match="'dt_min' must be less than 'dt_max'"):
        TimeDeltaSamplerParams(dt_min=10, dt_max=1, unit="D", reference_column_name="datetime")


def test_uuid_sampler_params():
    params = UUIDSamplerParams(prefix="test", short_form=True, uppercase=True)
    assert params.prefix == "test"
    assert params.short_form is True
    assert params.uppercase is True
    assert params.last_index == 8

    assert UUIDSamplerParams(prefix="invalid", short_form=False).last_index == 32


def test_person_sampler_params(stub_person_sampler_params):
    assert stub_person_sampler_params.locale == "en_US"
    assert stub_person_sampler_params.sex == "Female"
    assert stub_person_sampler_params.city == "New York"
    assert stub_person_sampler_params.age_range == [18, 30]
    assert stub_person_sampler_params.select_field_values is None
    assert stub_person_sampler_params.with_synthetic_personas is False
    assert stub_person_sampler_params.generator_kwargs == [
        "sex",
        "city",
        "age_range",
        "select_field_values",
        "with_synthetic_personas",
    ]
    assert stub_person_sampler_params.people_gen_key == "en_US"

    # update with synthetic personas
    stub_person_sampler_params.with_synthetic_personas = True
    assert stub_person_sampler_params.people_gen_key == "en_US_with_personas"


def test_person_sampler_age_range_validation():
    with pytest.raises(
        ValidationError,
        match="The first integer \\(min age\\) must be greater than or equal to 0, but the first integer provided was -1",
    ):
        PersonSamplerParams(locale="en_US", age_range=[-1, 15])
    with pytest.raises(
        ValidationError,
        match="The second integer \\(max age\\) must be less than or equal to 114, but the second integer provided was 1000",
    ):
        PersonSamplerParams(locale="en_US", age_range=[18, 1000])
    with pytest.raises(
        ValidationError, match="The first integer \\(min age\\) must be less than the second integer \\(max age\\)"
    ):
        PersonSamplerParams(locale="en_US", age_range=[18, 17])


def test_person_sampler_locale_validation():
    with pytest.raises(
        ValidationError,
        match="Person sampling from managed datasets is only supported for the following locales:",
    ):
        PersonSamplerParams(locale="invalid", age_range=[18, 30])


def test_person_sampler_state_validation():
    # state parameter has been replaced with select_field_values
    # Testing that select_field_values works correctly
    person_sampler = PersonSamplerParams(
        locale="en_US", select_field_values={"state": ["NY", "CA"]}, age_range=[18, 30]
    )
    assert person_sampler.select_field_values == {"state": ["NY", "CA"]}


def test_person_sampler_with_synthetic_personas_validation():
    # PersonSamplerParams now only supports locales with managed datasets
    # so trying to use a non-managed locale will fail before with_synthetic_personas is even validated
    with pytest.raises(
        ValidationError,
        match="Person sampling from managed datasets is only supported for the following locales:",
    ):
        PersonSamplerParams(locale="en_GB", with_synthetic_personas=True, age_range=[18, 30])


def test_is_numerical_sampler_type():
    assert is_numerical_sampler_type(SamplerType.BERNOULLI_MIXTURE) is True
    assert is_numerical_sampler_type(SamplerType.BERNOULLI) is True
    assert is_numerical_sampler_type(SamplerType.BINOMIAL) is True
    assert is_numerical_sampler_type(SamplerType.GAUSSIAN) is True
    assert is_numerical_sampler_type(SamplerType.POISSON) is True
    assert is_numerical_sampler_type(SamplerType.SCIPY) is True
    assert is_numerical_sampler_type(SamplerType.UNIFORM) is True
    assert is_numerical_sampler_type(SamplerType.CATEGORY) is False
    assert is_numerical_sampler_type(SamplerType.DATETIME) is False
    assert is_numerical_sampler_type(SamplerType.TIMEDELTA) is False
    assert is_numerical_sampler_type(SamplerType.UUID) is False
    assert is_numerical_sampler_type(SamplerType.PERSON) is False
    assert is_numerical_sampler_type(SamplerType.SUBCATEGORY) is False
