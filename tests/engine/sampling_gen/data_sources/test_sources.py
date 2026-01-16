# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from data_designer.config.sampler_params import (
    BernoulliSamplerParams,
    CategorySamplerParams,
    DatetimeSamplerParams,
    GaussianSamplerParams,
    PersonSamplerParams,
    SamplerType,
    ScipySamplerParams,
    SubcategorySamplerParams,
    TimeDeltaSamplerParams,
    UniformSamplerParams,
    UUIDSamplerParams,
)
from data_designer.engine.sampling_gen.data_sources.base import (
    DataSource,
    DatetimeFormatMixin,
    PassthroughMixin,
    Sampler,
    ScipyStatsSampler,
    TypeConversionMixin,
)
from data_designer.engine.sampling_gen.data_sources.errors import InvalidSamplerParamsError
from data_designer.engine.sampling_gen.data_sources.sources import (
    BernoulliSampler,
    CategorySampler,
    DatetimeSampler,
    GaussianSampler,
    PersonSampler,
    SamplerRegistry,
    ScipySampler,
    SubcategorySampler,
    TimeDeltaSampler,
    UniformSampler,
    UUIDSampler,
    load_sampler,
)
from data_designer.lazy_heavy_imports import np, pd

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


@pytest.fixture
def stub_people_gen():
    mock_gen = Mock()
    mock_gen.generate.return_value = [
        {"first_name": "John", "last_name": "Doe"},
        {"first_name": "Jane", "last_name": "Smith"},
    ]
    return mock_gen


def test_data_source_get_param_type():
    assert hasattr(DataSource, "get_param_type")


def test_data_source_get_required_column_names():
    assert hasattr(DataSource, "get_required_column_names")


def test_sampler_recast_types_if_needed():
    assert hasattr(Sampler, "_recast_types_if_needed")


def test_sampler_inject_data_column_empty_index():
    assert hasattr(Sampler, "inject_data_column")


def test_scipy_stats_sampler_sample_method():
    assert hasattr(ScipyStatsSampler, "sample")


def test_passthrough_mixin_preproc_passthrough():
    series = pd.Series([1, 2, 3])
    result = PassthroughMixin.preproc(series, "int")
    pd.testing.assert_series_equal(result, series)


def test_passthrough_mixin_postproc_passthrough():
    series = pd.Series([1, 2, 3])
    result = PassthroughMixin.postproc(series, "int")
    pd.testing.assert_series_equal(result, series)


def test_passthrough_mixin_validate_data_conversion_passthrough():
    PassthroughMixin.validate_data_conversion("int")
    PassthroughMixin.validate_data_conversion(None)


def test_type_conversion_mixin_preproc_type_conversion():
    series = pd.Series([1.5, 2.7, 3.2])

    result = TypeConversionMixin.preproc(series, "int")
    expected = pd.Series([2, 3, 3], dtype="int64")
    pd.testing.assert_series_equal(result, expected)

    result = TypeConversionMixin.preproc(series, "str")
    expected = pd.Series(["1.5", "2.7", "3.2"], dtype="str")
    pd.testing.assert_series_equal(result, expected)


def test_type_conversion_mixin_postproc_type_conversion():
    series = pd.Series([1.5, 2.7, 3.2])

    result = TypeConversionMixin.postproc(series, "int")
    expected = pd.Series([2, 3, 3], dtype="int64")
    pd.testing.assert_series_equal(result, expected)


def test_type_conversion_mixin_validate_data_conversion_valid():
    TypeConversionMixin.validate_data_conversion("float")
    TypeConversionMixin.validate_data_conversion("int")
    TypeConversionMixin.validate_data_conversion("str")
    TypeConversionMixin.validate_data_conversion(None)


def test_type_conversion_mixin_validate_data_conversion_invalid():
    with pytest.raises(ValueError, match="Invalid `convert_to` value"):
        TypeConversionMixin.validate_data_conversion("invalid")


def test_datetime_format_mixin_preproc_datetime():
    series = pd.Series(pd.date_range("2023-01-01", periods=3))
    result = DatetimeFormatMixin.preproc(series, "%Y-%m-%d")
    pd.testing.assert_series_equal(result, series)


def test_datetime_format_mixin_postproc_datetime_formatting():
    series = pd.Series(pd.date_range("2023-01-01", periods=3))
    result = DatetimeFormatMixin.postproc(series, "%Y-%m-%d")
    expected = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"], dtype="str")
    pd.testing.assert_series_equal(result, expected)


def test_datetime_format_mixin_validate_data_conversion_valid_format():
    DatetimeFormatMixin.validate_data_conversion("%Y-%m-%d")
    DatetimeFormatMixin.validate_data_conversion(None)


def test_datetime_format_mixin_validate_data_conversion_invalid_format():
    with pytest.raises(ValueError, match="Invalid datetime format"):
        DatetimeFormatMixin.validate_data_conversion("invalid_format")


def test_sampler_registry_register_decorator():
    class TestSampler(DataSource):
        pass

    SamplerRegistry.register("test")(TestSampler)
    assert "test" in SamplerRegistry._registry
    assert SamplerRegistry._registry["test"] == TestSampler


def test_sampler_registry_get_sampler():
    class TestSampler(DataSource):
        pass

    SamplerRegistry.register("test")(TestSampler)
    result = SamplerRegistry.get_sampler("TEST")
    assert result == TestSampler


def test_sampler_registry_is_registered():
    class TestSampler(DataSource):
        pass

    SamplerRegistry.register("test")(TestSampler)
    assert SamplerRegistry.is_registered("test")
    assert not SamplerRegistry.is_registered("nonexistent")


def test_sampler_registry_validate_sampler_type_string():
    class TestSampler(DataSource):
        pass

    SamplerRegistry.register("test")(TestSampler)
    result = SamplerRegistry.validate_sampler_type("test")
    assert result == TestSampler


def test_sampler_registry_validate_sampler_type_class():
    class TestSampler(DataSource):
        pass

    result = SamplerRegistry.validate_sampler_type(TestSampler)
    assert result == TestSampler


def test_sampler_registry_validate_sampler_type_invalid_string():
    with pytest.raises(ValueError, match="Sampler type `invalid` not found"):
        SamplerRegistry.validate_sampler_type("invalid")


def test_sampler_registry_validate_sampler_type_invalid_class():
    class NotDataSource:
        pass

    with pytest.raises(ValueError, match="is not a subclass of `DataSource`"):
        SamplerRegistry.validate_sampler_type(NotDataSource)


def test_subcategory_sampler_get_required_column_names():
    params = SubcategorySamplerParams(category="test_col", values={"A": ["1", "2"], "B": ["3", "4"]})
    sampler = SubcategorySampler(params=params)
    assert sampler.get_required_column_names() == ("test_col",)


def test_subcategory_sampler_inject_data_column(stub_sample_dataframe):
    params = SubcategorySamplerParams(category="category", values={"A": ["1", "2"], "B": ["3", "4"]})
    sampler = SubcategorySampler(params=params)

    result = sampler.inject_data_column(stub_sample_dataframe, "new_col", index=[0, 1, 2, 3])
    assert "new_col" in result.columns
    assert len(result) == 4


def test_category_sampler_sample():
    params = CategorySamplerParams(values=["A", "B", "C"], weights=[0.5, 0.3, 0.2])
    sampler = CategorySampler(params=params, random_state=42)

    result = sampler.sample(10)
    assert len(result) == 10
    assert all(val in ["A", "B", "C"] for val in result)


def test_datetime_sampler_sample():
    params = DatetimeSamplerParams(start="2023-01-01", end="2023-12-31", unit="D")
    sampler = DatetimeSampler(params=params, random_state=42)

    result = sampler.sample(5)
    assert len(result) == 5
    assert all(isinstance(val, np.datetime64) for val in result)


def test_person_sampler_setup_with_generator(stub_people_gen):
    params = PersonSamplerParams()
    people_gen_resource = {"en_US": stub_people_gen}

    sampler = PersonSampler(params=params, people_gen_resource=people_gen_resource)
    assert sampler._generator == stub_people_gen


def test_person_sampler_setup_without_generator():
    params = PersonSamplerParams()

    sampler = PersonSampler(params=params)
    assert sampler._generator is None


def test_person_sampler_sample_with_generator(stub_people_gen):
    params = PersonSamplerParams()

    sampler = PersonSampler(params=params)
    sampler.set_generator(stub_people_gen)

    result = sampler.sample(2)
    assert len(result) == 2


def test_person_sampler_sample_without_generator():
    params = PersonSamplerParams()

    sampler = PersonSampler(params=params)

    with pytest.raises(ValueError, match="Generator not set"):
        sampler.sample(2)


def test_time_delta_sampler_get_required_column_names():
    params = TimeDeltaSamplerParams(reference_column_name="date_col", dt_min=1, dt_max=30, unit="D")
    sampler = TimeDeltaSampler(params=params)
    assert sampler.get_required_column_names() == ("date_col",)


def test_time_delta_sampler_sample():
    params = TimeDeltaSamplerParams(reference_column_name="date_col", dt_min=1, dt_max=30, unit="D")
    sampler = TimeDeltaSampler(params=params, random_state=42)

    result = sampler.sample(5)
    assert len(result) == 5
    assert all(isinstance(val, np.timedelta64) for val in result)


def test_uuid_sampler_sample_basic():
    params = UUIDSamplerParams(prefix="TEST", short_form=True, uppercase=True)
    sampler = UUIDSampler(params=params, random_state=42)

    result = sampler.sample(3)
    assert len(result) == 3
    assert all(isinstance(val, str) for val in result)
    assert all(val.startswith("TEST") for val in result)


def test_uuid_sampler_sample_no_duplicates():
    params = UUIDSamplerParams(prefix="", short_form=True, uppercase=False)
    sampler = UUIDSampler(params=params, random_state=42)

    result = sampler.sample(10)
    assert len(result) == 10
    assert len(set(result)) == 10


def test_scipy_sampler_distribution_property():
    params = ScipySamplerParams(dist_name="norm", dist_params={"loc": 0, "scale": 1})
    sampler = ScipySampler(params=params)

    dist = sampler.distribution
    assert hasattr(dist, "rvs")


def test_scipy_sampler_sample():
    params = ScipySamplerParams(dist_name="norm", dist_params={"loc": 0, "scale": 1})
    sampler = ScipySampler(params=params, random_state=42)

    result = sampler.sample(10)
    assert len(result) == 10
    assert all(isinstance(val, (int, float)) for val in result)


def test_scipy_sampler_validate_invalid_distribution():
    params = ScipySamplerParams(dist_name="nonexistent", dist_params={})

    with pytest.raises(InvalidSamplerParamsError):
        ScipySampler(params=params)


def test_bernoulli_sampler():
    params = BernoulliSamplerParams(p=0.5)
    sampler = BernoulliSampler(params=params, random_state=42)
    result = sampler.sample(10)
    assert len(result) == 10
    assert all(val in [0, 1] for val in result)


def test_gaussian_sampler():
    params = GaussianSamplerParams(mean=0, stddev=1)
    sampler = GaussianSampler(params=params, random_state=42)
    result = sampler.sample(10)
    assert len(result) == 10
    assert all(isinstance(val, (int, float)) for val in result)


def test_uniform_sampler():
    params = UniformSamplerParams(low=0, high=10)
    sampler = UniformSampler(params=params, random_state=42)
    result = sampler.sample(10)
    assert len(result) == 10
    assert all(0 <= val <= 10 for val in result)


def test_load_sampler():
    sampler = load_sampler(SamplerType.GAUSSIAN, mean=0, stddev=1)
    assert isinstance(sampler, GaussianSampler)


def test_load_sampler_invalid_type():
    with pytest.raises(ValueError):
        load_sampler("invalid_type")
