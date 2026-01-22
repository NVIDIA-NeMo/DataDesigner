# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from data_designer.config.analysis.column_statistics import (
    CategoricalDistribution,
    CategoricalHistogramData,
    ColumnDistributionType,
    ExpressionColumnStatistics,
    GeneralColumnStatistics,
    LLMCodeColumnStatistics,
    LLMJudgedColumnStatistics,
    LLMStructuredColumnStatistics,
    LLMTextColumnStatistics,
    MissingValue,
    NumericalDistribution,
    SamplerColumnStatistics,
    SamplerType,
    ValidationColumnStatistics,
)
from data_designer.lazy_heavy_imports import np, pd

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


@pytest.fixture
def stub_categorical_histogram_data():
    return CategoricalHistogramData(
        categories=["A", "B", "C"],
        counts=[3, 2, 1],
    )


@pytest.fixture
def stub_categorical_distribution(stub_categorical_histogram_data):
    return CategoricalDistribution(
        most_common_value="A",
        least_common_value="C",
        histogram=stub_categorical_histogram_data,
    )


@pytest.fixture
def stub_general_stats_args_with_missing_values():
    return {
        "column_name": "test",
        "num_records": MissingValue.CALCULATION_FAILED,
        "num_null": MissingValue.CALCULATION_FAILED,
        "num_unique": MissingValue.CALCULATION_FAILED,
        "pyarrow_dtype": "string",
        "simple_dtype": "str",
    }


@pytest.fixture
def stub_general_stats_args_with_valid_values():
    return {
        "column_name": "test",
        "num_records": 100,
        "num_null": 10,
        "num_unique": 10,
        "pyarrow_dtype": "string",
        "simple_dtype": "str",
    }


@pytest.mark.parametrize(
    "general_column_statistics_based_class,column_type",
    [
        (GeneralColumnStatistics, "general"),
        (ExpressionColumnStatistics, "expression"),
    ],
)
def test_general_column_statistics_with_missing_values(
    stub_general_stats_args_with_missing_values, general_column_statistics_based_class, column_type
):
    general_column_statistics = general_column_statistics_based_class(
        **stub_general_stats_args_with_missing_values,
    )
    assert general_column_statistics.percent_null == MissingValue.CALCULATION_FAILED
    assert general_column_statistics.percent_unique == MissingValue.CALCULATION_FAILED
    assert general_column_statistics.column_type == column_type
    assert general_column_statistics.create_report_row_data() == {
        "column name": "test",
        "number unique values": "--",
        "data type": "str",
    }


@pytest.mark.parametrize(
    "general_column_statistics_based_class,column_type",
    [
        (GeneralColumnStatistics, "general"),
        (ExpressionColumnStatistics, "expression"),
    ],
)
def test_general_column_statistics_with_valid_values(
    stub_general_stats_args_with_valid_values, general_column_statistics_based_class, column_type
):
    general_column_statistics = general_column_statistics_based_class(
        **stub_general_stats_args_with_valid_values,
    )
    assert general_column_statistics.percent_null == 10.0
    assert general_column_statistics.percent_unique == 10.0
    assert general_column_statistics.column_type == column_type
    assert general_column_statistics.create_report_row_data() == {
        "column name": "test",
        "number unique values": "10 (10.0%)",
        "data type": "str",
    }


@pytest.mark.parametrize(
    "llm_text_column_statistics_based_class,column_type",
    [
        (LLMTextColumnStatistics, "llm-text"),
        (LLMCodeColumnStatistics, "llm-code"),
        (LLMStructuredColumnStatistics, "llm-structured"),
        (LLMJudgedColumnStatistics, "llm-judge"),
    ],
)
def test_llm_text_column_statistics_with_missing_values(
    stub_general_stats_args_with_missing_values, llm_text_column_statistics_based_class, column_type
):
    llm_text_column_statistics = llm_text_column_statistics_based_class(
        **stub_general_stats_args_with_missing_values,
        output_tokens_mean=MissingValue.CALCULATION_FAILED,
        output_tokens_median=MissingValue.CALCULATION_FAILED,
        output_tokens_stddev=MissingValue.CALCULATION_FAILED,
        input_tokens_mean=MissingValue.CALCULATION_FAILED,
        input_tokens_median=MissingValue.CALCULATION_FAILED,
        input_tokens_stddev=MissingValue.CALCULATION_FAILED,
    )
    assert llm_text_column_statistics.column_type == column_type
    assert llm_text_column_statistics.create_report_row_data() == {
        "column name": "test",
        "number unique values": "--",
        "data type": "str",
        "prompt tokens\nper record": "--",
        "completion tokens\nper record": "--",
    }


@pytest.mark.parametrize(
    "llm_text_column_statistics_based_class,column_type",
    [
        (LLMTextColumnStatistics, "llm-text"),
        (LLMCodeColumnStatistics, "llm-code"),
        (LLMStructuredColumnStatistics, "llm-structured"),
        (LLMJudgedColumnStatistics, "llm-judge"),
    ],
)
def test_llm_text_column_statistics_with_valid_values(
    stub_general_stats_args_with_valid_values, llm_text_column_statistics_based_class, column_type
):
    llm_text_column_statistics = llm_text_column_statistics_based_class(
        **stub_general_stats_args_with_valid_values,
        output_tokens_mean=150.0,
        output_tokens_median=150.0,
        output_tokens_stddev=25.2,
        input_tokens_mean=50.0,
        input_tokens_median=50.0,
        input_tokens_stddev=10.0,
    )
    assert llm_text_column_statistics.column_type == column_type
    assert llm_text_column_statistics.create_report_row_data() == {
        "column name": "test",
        "number unique values": "10 (10.0%)",
        "data type": "str",
        "prompt tokens\nper record": "50.0 +/- 10.0",
        "completion tokens\nper record": "150.0 +/- 25.2",
    }


def test_sampler_column_statistics(stub_general_stats_args_with_valid_values, stub_categorical_distribution):
    sampler_column_statistics = SamplerColumnStatistics(
        **stub_general_stats_args_with_valid_values,
        sampler_type=SamplerType.CATEGORY,
        distribution_type=ColumnDistributionType.CATEGORICAL,
        distribution=stub_categorical_distribution,
    )
    assert sampler_column_statistics.column_type == "sampler"
    assert isinstance(sampler_column_statistics.distribution, CategoricalDistribution)
    assert sampler_column_statistics.create_report_row_data() == {
        "column name": "test",
        "number unique values": "10 (10.0%)",
        "data type": "str",
        "sampler type": "category",
    }


def test_validation_column_statistics_with_missing_values(stub_general_stats_args_with_missing_values):
    validation_column_statistics = ValidationColumnStatistics(
        **stub_general_stats_args_with_missing_values,
        num_valid_records=MissingValue.CALCULATION_FAILED,
    )
    assert validation_column_statistics.column_type == "validation"
    assert validation_column_statistics.create_report_row_data() == {
        "column name": "test",
        "number unique values": "--",
        "data type": "str",
        "percent valid": "--",
    }


def test_validation_column_statistics_with_valid_values(stub_general_stats_args_with_valid_values):
    validation_column_statistics = ValidationColumnStatistics(
        **stub_general_stats_args_with_valid_values,
        num_valid_records=100,
    )
    assert validation_column_statistics.column_type == "validation"
    assert validation_column_statistics.create_report_row_data() == {
        "column name": "test",
        "number unique values": "10 (10.0%)",
        "data type": "str",
        "percent valid": "100.0%",
    }


def test_categorical_histogram_data():
    # test construction
    categorical_histogram_data = CategoricalHistogramData(
        categories=[np.int64(1), np.int64(2), np.int64(3), np.int64(4)],
        counts=[3.0, 2.0, 1.0, 0.0],
    )
    assert categorical_histogram_data.categories == [1, 2, 3, 4]
    assert categorical_histogram_data.counts == [3, 2, 1, 0]

    # test from pd series
    categorical_histogram_data = CategoricalHistogramData.from_series(
        pd.Series(
            [np.float16(1.0), np.float16(1.0), np.float16(2.0), np.float16(3.0), np.float16(3.0), np.float16(1.0)]
        )
    )
    assert categorical_histogram_data.categories == [1, 3, 2]
    assert categorical_histogram_data.counts == [3, 2, 1]


def test_categorical_distribution(stub_categorical_histogram_data):
    categorical_distribution = CategoricalDistribution(
        most_common_value=np.int8(1),
        least_common_value=np.int8(3),
        histogram=stub_categorical_histogram_data,
    )
    assert categorical_distribution.most_common_value == 1
    assert categorical_distribution.least_common_value == 3

    categorical_distribution = CategoricalDistribution(
        most_common_value=1.0,
        least_common_value=3.0,
        histogram=stub_categorical_histogram_data,
    )
    assert categorical_distribution.most_common_value == "1.0"
    assert categorical_distribution.least_common_value == "3.0"

    # test from series
    categorical_distribution = CategoricalDistribution.from_series(
        pd.Series([np.int8(1), np.int8(2), np.int8(3), np.int8(3), np.int8(2), np.int8(3)])
    )
    assert categorical_distribution.most_common_value == 3
    assert categorical_distribution.least_common_value == 1
    assert categorical_distribution.histogram.categories == [3, 2, 1]
    assert categorical_distribution.histogram.counts == [3, 2, 1]


def test_numerical_distribution():
    numerical_distribution = NumericalDistribution(
        min=np.float16(1.0),
        max=np.int8(2.0),
        mean=np.int8(3),
        stddev=np.int8(4),
        median=5,
    )
    assert numerical_distribution.min == 1
    assert numerical_distribution.max == 2
    assert numerical_distribution.mean == 3.0
    assert numerical_distribution.stddev == 4.0
    assert numerical_distribution.median == 5.0

    assert isinstance(numerical_distribution.min, float)
    assert isinstance(numerical_distribution.max, int)
    assert isinstance(numerical_distribution.mean, float)
    assert isinstance(numerical_distribution.stddev, float)
    assert isinstance(numerical_distribution.median, float)

    # test from series
    numerical_distribution = NumericalDistribution.from_series(pd.Series([1, 2, 3, 4, 5, pd.NA]))
    assert numerical_distribution.min == 1
    assert numerical_distribution.max == 5
    assert numerical_distribution.mean == 3.0
    assert numerical_distribution.stddev == 1.58
    assert numerical_distribution.median == 3.0
