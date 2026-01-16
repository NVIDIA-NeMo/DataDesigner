# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import cycle
from typing import TYPE_CHECKING

import pytest

from data_designer.config.analysis.column_statistics import (
    CategoricalDistribution,
    CategoricalHistogramData,
    ColumnDistributionType,
    MissingValue,
    NumericalDistribution,
)
from data_designer.config.column_configs import LLMTextColumnConfig
from data_designer.config.utils.numerical_helpers import prepare_number_for_reporting
from data_designer.engine.analysis.utils.column_statistics_calculations import (
    calculate_column_distribution,
    calculate_general_column_info,
    calculate_input_token_stats,
    calculate_output_token_stats,
    calculate_validation_column_info,
    convert_pyarrow_dtype_to_simple_dtype,
    ensure_boolean,
    ensure_hashable,
)
from data_designer.lazy_heavy_imports import np, pa, pd

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import pyarrow as pa


@pytest.fixture
def stub_column_config():
    return LLMTextColumnConfig(
        name="test_column",
        prompt="Test prompt",
        system_prompt="System prompt",
        model_alias="test_model_alias",
    )


@pytest.fixture
def stub_df_responses():
    return pd.DataFrame({"test_column": ["short", "this is a longer response"]})


@pytest.fixture
def stub_df_code_validation():
    return pd.DataFrame(
        {"test_column": [{"is_valid": True}, {"is_valid": False}, {"is_valid": True}, {"is_valid": True}]}
    )


def test_categorical_histogram_data_from_series():
    series = pd.Series(["A", "B", "A", "C", "B", "A"])
    histogram = CategoricalHistogramData.from_series(series)
    assert histogram.categories == ["A", "B", "C"]
    assert histogram.counts == [3, 2, 1]

    series_numpy = pd.Series([np.int64(1), np.int64(2), np.int64(1), np.float64(3.0)])
    histogram_numpy = CategoricalHistogramData.from_series(series_numpy)
    assert histogram_numpy.categories == [1, 2, 3.0]
    assert histogram_numpy.counts == [2, 1, 1]
    assert all(isinstance(x, (int, float)) for x in histogram_numpy.categories)
    assert all(isinstance(x, int) for x in histogram_numpy.counts)


def test_categorical_distribution_from_series():
    series = pd.Series(["A", "B", "A", "C", "B", "A"])
    distribution = CategoricalDistribution.from_series(series)
    assert distribution.most_common_value == "A"
    assert distribution.least_common_value == "C"
    assert isinstance(distribution.histogram, CategoricalHistogramData)
    assert distribution.histogram.categories == ["A", "B", "C"]
    assert distribution.histogram.counts == [3, 2, 1]

    series_numpy = pd.Series([np.int64(1), np.int64(2), np.int64(1)])
    distribution_numpy = CategoricalDistribution.from_series(series_numpy)
    assert distribution_numpy.most_common_value == 1
    assert distribution_numpy.least_common_value == 2
    assert isinstance(distribution_numpy.most_common_value, int)
    assert isinstance(distribution_numpy.least_common_value, int)


def test_numerical_distribution_from_series():
    series = pd.Series([1, 2, 3, 4, 5])
    distribution = NumericalDistribution.from_series(series)
    assert distribution.min == 1
    assert distribution.max == 5
    assert distribution.mean == 3.0
    assert distribution.stddev == pytest.approx(1.58, abs=0.01)
    assert distribution.median == 3.0

    series_with_nan = pd.Series([1, 2, np.nan, 4, 5])
    distribution_with_nan = NumericalDistribution.from_series(series_with_nan)
    assert distribution_with_nan.min == 1
    assert distribution_with_nan.max == 5
    assert distribution_with_nan.mean == 3.0
    assert distribution_with_nan.median == 3.0

    distribution_numpy = NumericalDistribution(
        min=np.int64(1),
        max=np.float64(5.0),
        mean=np.float64(3.0),
        stddev=np.float64(1.5),
        median=np.float64(3.0),
    )
    assert distribution_numpy.min == 1
    assert distribution_numpy.max == 5.0
    assert distribution_numpy.mean == 3.0
    assert distribution_numpy.stddev == 1.5
    assert distribution_numpy.median == 3.0
    assert isinstance(distribution_numpy.min, int)
    assert isinstance(distribution_numpy.max, float)


def test_calculate_column_distribution():
    column_name = "test_column"

    df_categorical = pd.DataFrame({"test_column": ["A", "B", "A", "C", "B", "A"]})
    result = calculate_column_distribution(column_name, df_categorical, ColumnDistributionType.CATEGORICAL)
    assert result["distribution_type"] == ColumnDistributionType.CATEGORICAL
    assert isinstance(result["distribution"], CategoricalDistribution)
    assert result["distribution"].most_common_value == "A"

    df_numerical = pd.DataFrame({"test_column": [1, 2, 3, 4, 5]})
    result = calculate_column_distribution(column_name, df_numerical, ColumnDistributionType.NUMERICAL)
    assert result["distribution_type"] == ColumnDistributionType.NUMERICAL
    assert isinstance(result["distribution"], NumericalDistribution)
    assert result["distribution"].min == 1
    assert result["distribution"].max == 5

    column_name = "nonexistent_column"
    df_other = pd.DataFrame({"other_column": [1, 2, 3]})
    result = calculate_column_distribution(column_name, df_other, ColumnDistributionType.CATEGORICAL)
    assert result["distribution_type"] == ColumnDistributionType.UNKNOWN
    assert result["distribution"] == MissingValue.CALCULATION_FAILED


def test_calculate_general_column_info(stub_df_with_mixed_column_types):
    column_name = "int_with_nulls_column"
    result = calculate_general_column_info(column_name, stub_df_with_mixed_column_types)
    assert result["num_records"] == 5
    assert result["num_null"] == 2
    assert result["num_unique"] == 3
    assert result["simple_dtype"] == "int"
    assert "pyarrow_dtype" in result

    column_name = "string_column"
    result = calculate_general_column_info(column_name, stub_df_with_mixed_column_types)
    assert result["num_records"] == 5
    assert result["num_null"] == 0
    assert result["num_unique"] == 5
    assert result["simple_dtype"] == "string"
    assert "pyarrow_dtype" in result

    column_name = "float_column"
    result = calculate_general_column_info(column_name, stub_df_with_mixed_column_types)
    assert result["num_records"] == 5
    assert result["num_null"] == 0
    assert result["num_unique"] == 5
    assert result["simple_dtype"] == "float"
    assert "pyarrow_dtype" in result

    column_name = "nonexistent_column"
    result = calculate_general_column_info(column_name, stub_df_with_mixed_column_types)
    assert result["num_records"] == MissingValue.CALCULATION_FAILED
    assert result["num_null"] == MissingValue.CALCULATION_FAILED
    assert result["num_unique"] == MissingValue.CALCULATION_FAILED


def test_calculate_input_token_stats(mock_prompt_renderer_render, stub_column_config, stub_df_responses):
    prompt_cycle = cycle(["System prompt", "Test prompt"])
    mock_prompt_renderer_render.side_effect = lambda *args, **kwargs: next(prompt_cycle)

    result = calculate_input_token_stats(stub_column_config, stub_df_responses)
    assert "input_tokens_mean" in result
    assert "input_tokens_stddev" in result
    assert "input_tokens_median" in result
    assert isinstance(result["input_tokens_mean"], float)
    assert isinstance(result["input_tokens_stddev"], float)
    assert isinstance(result["input_tokens_median"], float)

    mock_prompt_renderer_render.side_effect = Exception("Test error")
    result = calculate_input_token_stats(stub_column_config, stub_df_responses)
    assert result["input_tokens_mean"] == MissingValue.CALCULATION_FAILED
    assert result["input_tokens_stddev"] == MissingValue.CALCULATION_FAILED
    assert result["input_tokens_median"] == MissingValue.CALCULATION_FAILED


def test_calculate_output_token_stats(stub_column_config, stub_df_responses):
    result = calculate_output_token_stats(stub_column_config, stub_df_responses)
    assert "output_tokens_mean" in result
    assert "output_tokens_stddev" in result
    assert "output_tokens_median" in result
    assert isinstance(result["output_tokens_mean"], float)
    assert isinstance(result["output_tokens_stddev"], float)
    assert isinstance(result["output_tokens_median"], float)

    stub_column_config.name = "nonexistent_column"
    result = calculate_output_token_stats(stub_column_config, stub_df_responses)
    assert result["output_tokens_mean"] == MissingValue.CALCULATION_FAILED
    assert result["output_tokens_stddev"] == MissingValue.CALCULATION_FAILED
    assert result["output_tokens_median"] == MissingValue.CALCULATION_FAILED


def test_calculate_validation_column_info(stub_column_config, stub_df_code_validation):
    result = calculate_validation_column_info(stub_column_config.name, stub_df_code_validation)
    assert result["num_valid_records"] == 3

    result = calculate_validation_column_info("nonexistent_column", stub_df_code_validation)
    assert result["num_valid_records"] == MissingValue.CALCULATION_FAILED


def test_convert_pyarrow_dtype_to_simple_dtype():
    assert convert_pyarrow_dtype_to_simple_dtype(pa.int64()) == "int"
    assert convert_pyarrow_dtype_to_simple_dtype(pa.int32()) == "int"
    assert convert_pyarrow_dtype_to_simple_dtype(pa.int16()) == "int"
    assert convert_pyarrow_dtype_to_simple_dtype(pa.float64()) == "float"
    assert convert_pyarrow_dtype_to_simple_dtype(pa.float32()) == "float"
    assert convert_pyarrow_dtype_to_simple_dtype(pa.string()) == "string"
    assert convert_pyarrow_dtype_to_simple_dtype(pa.timestamp("s")) == "timestamp"
    assert convert_pyarrow_dtype_to_simple_dtype(pa.time32("s")) == "time"
    assert convert_pyarrow_dtype_to_simple_dtype(pa.date32()) == "date"

    list_type = pa.list_(pa.string())
    assert convert_pyarrow_dtype_to_simple_dtype(list_type) == "list[string]"

    nested_list_type = pa.list_(pa.list_(pa.int64()))
    assert convert_pyarrow_dtype_to_simple_dtype(nested_list_type) == "list[list[int]]"

    struct_type = pa.struct([("field1", pa.string()), ("field2", pa.int64())])
    assert convert_pyarrow_dtype_to_simple_dtype(struct_type) == "dict"

    unknown_type = pa.binary()
    assert convert_pyarrow_dtype_to_simple_dtype(unknown_type) == str(unknown_type)


def test_prepare_number_for_reporting():
    assert prepare_number_for_reporting(5, int) == 5
    assert isinstance(prepare_number_for_reporting(5, int), int)

    assert prepare_number_for_reporting(3.14159, float, precision=2) == 3.14
    assert isinstance(prepare_number_for_reporting(3.14159, float, precision=2), float)

    assert prepare_number_for_reporting(np.int64(5), int) == 5
    assert not isinstance(prepare_number_for_reporting(np.int64(5), int), np.int64)

    assert prepare_number_for_reporting(np.float64(3.14159), float, precision=2) == 3.14
    assert not isinstance(prepare_number_for_reporting(np.float64(3.14159), float, precision=2), np.float64)

    assert prepare_number_for_reporting(3.14159, float, precision=3) == 3.142


def test_ensure_hashable():
    assert ensure_hashable(5) == 5
    assert ensure_hashable(3.14) == 3.14
    assert ensure_hashable(True) is True
    assert ensure_hashable(None) is None
    assert ensure_hashable("hello") == "hello"

    result = ensure_hashable({"b": 2, "a": 1})
    assert isinstance(result, str)
    assert "a" in result and "b" in result

    assert ensure_hashable([3, 1, 2]) == "[1, 2, 3]"

    result = ensure_hashable({"a": [1, 2], "b": {"c": 3}})
    assert isinstance(result, str)
    assert "a" in result and "b" in result

    with pytest.raises(TypeError):
        hash(np.array([1, 2, 3]))
    assert isinstance(hash(ensure_hashable(np.array([1, 2, 3]))), int)


def test_ensure_boolean():
    assert ensure_boolean(True) is True
    assert ensure_boolean(False) is False
    assert ensure_boolean(np.bool_(True)) is True

    assert ensure_boolean(1) is True
    assert ensure_boolean(0) is False
    assert ensure_boolean(1.0) is True
    assert ensure_boolean(0.0) is False
    assert ensure_boolean(np.int64(1)) is True
    assert ensure_boolean(np.float64(0.0)) is False

    assert ensure_boolean("true") is True
    assert ensure_boolean("false") is False
    assert ensure_boolean("TRUE") is True
    assert ensure_boolean("FALSE") is False
    assert ensure_boolean(np.str_("true")) is True

    with pytest.raises(ValueError):
        ensure_boolean("invalid")
    with pytest.raises(ValueError):
        ensure_boolean(2)
    with pytest.raises(ValueError):
        ensure_boolean(1.5)


def test_calculate_general_column_info_dtype_detection():
    """Test dtype detection with PyArrow backend (preferred path)."""
    df_pyarrow = pa.Table.from_pydict(
        {"int_col": [1, 2, 3], "str_col": ["a", "b", "c"], "float_col": [1.1, 2.2, 3.3]}
    ).to_pandas(types_mapper=pd.ArrowDtype)

    result = calculate_general_column_info("int_col", df_pyarrow)
    assert result["simple_dtype"] == "int"
    assert result["pyarrow_dtype"] == "int64"

    result = calculate_general_column_info("str_col", df_pyarrow)
    assert result["simple_dtype"] == "string"
    assert "string" in result["pyarrow_dtype"]

    result = calculate_general_column_info("float_col", df_pyarrow)
    assert result["simple_dtype"] == "float"
    assert result["pyarrow_dtype"] == "double"


def test_calculate_general_column_info_dtype_detection_fallback():
    """Test dtype detection fallback when PyArrow backend unavailable (mixed types)."""
    df_mixed = pd.DataFrame({"mixed_col": [1, "two", 3.0, "four", 5]})

    result = calculate_general_column_info("mixed_col", df_mixed)
    assert result["simple_dtype"] == "int"
    assert result["pyarrow_dtype"] == "n/a"
    assert result["num_records"] == 5
    assert result["num_unique"] == 5


def test_calculate_general_column_info_edge_cases():
    """Test edge cases: nulls, empty columns, and all-null columns."""
    df_with_nulls = pd.DataFrame({"col_with_nulls": [None, None, 42.0, 43.0, 44.0]})
    result = calculate_general_column_info("col_with_nulls", df_with_nulls)
    assert result["simple_dtype"] == "float"
    assert result["num_null"] == 2
    assert result["num_unique"] == 3

    df_all_nulls = pd.DataFrame({"all_nulls": [None, None, None]})
    result = calculate_general_column_info("all_nulls", df_all_nulls)
    assert result["simple_dtype"] == MissingValue.CALCULATION_FAILED
    assert result["num_null"] == 3
    assert result["num_unique"] == 0

    df_empty = pd.DataFrame({"empty_col": []})
    result = calculate_general_column_info("empty_col", df_empty)
    assert result["num_records"] == 0
    assert result["num_null"] == 0
    assert result["simple_dtype"] == MissingValue.CALCULATION_FAILED
