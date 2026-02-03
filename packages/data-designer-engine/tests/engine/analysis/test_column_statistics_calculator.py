# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from data_designer.config.analysis.column_statistics import ColumnDistributionType
from data_designer.config.column_types import DataDesignerColumnType
from data_designer.config.sampler_params import SamplerType
from data_designer.engine.analysis.column_profilers.base import ColumnConfigWithDataFrame
from data_designer.engine.analysis.column_statistics import get_column_statistics_calculator
from data_designer.engine.analysis.utils.column_statistics_calculations import ensure_hashable
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd


def test_general_column_statistics(stub_df, column_configs):
    for column_config in column_configs:
        column_config_with_df = ColumnConfigWithDataFrame(column_config=column_config, df=stub_df)
        stats = get_column_statistics_calculator(column_config.column_type)(
            column_config_with_df=column_config_with_df
        ).calculate()

        assert stats.column_name == column_config.name
        assert stats.column_type == column_config.column_type
        assert stats.percent_null == 0.0
        assert stats.num_records == len(stub_df)

        _df = pd.DataFrame(stub_df[column_config.name].apply(ensure_hashable))
        assert stats.num_null == _df[column_config.name].isnull().sum()
        assert stats.num_unique == _df[column_config.name].nunique()


def test_llm_generated_column_statistics(stub_df, column_configs):
    for column_config in column_configs:
        if column_config.column_type in [
            DataDesignerColumnType.LLM_TEXT,
            DataDesignerColumnType.LLM_CODE,
            DataDesignerColumnType.LLM_STRUCTURED,
            DataDesignerColumnType.LLM_JUDGE,
        ]:
            column_config_with_df = ColumnConfigWithDataFrame(column_config=column_config, df=stub_df)
            stats = get_column_statistics_calculator(column_config.column_type)(
                column_config_with_df=column_config_with_df
            ).calculate()
            assert stats.column_name == column_config.name
            assert stats.column_type == column_config.column_type
            assert stats.num_records == len(stub_df)
            assert isinstance(stats.output_tokens_mean, float)
            assert isinstance(stats.output_tokens_stddev, float)
            assert isinstance(stats.input_tokens_mean, float)
            assert isinstance(stats.input_tokens_stddev, float)


def test_sampler_column_statistics(stub_df, column_configs):
    for column_config in column_configs:
        if column_config.column_type == DataDesignerColumnType.SAMPLER:
            column_config_with_df = ColumnConfigWithDataFrame(column_config=column_config, df=stub_df)
            stats = get_column_statistics_calculator(column_config.column_type)(
                column_config_with_df=column_config_with_df
            ).calculate()
            assert stats.column_name == column_config.name
            assert stats.column_type == column_config.column_type
            if column_config.sampler_type in [SamplerType.CATEGORY, SamplerType.SUBCATEGORY]:
                assert stats.distribution_type == ColumnDistributionType.CATEGORICAL
                assert hasattr(stats.distribution, "histogram")
                assert isinstance(stats.distribution.most_common_value, (int, str))
                assert isinstance(stats.distribution.least_common_value, (int, str))
            elif column_config.sampler_type != SamplerType.PERSON:
                assert stats.distribution_type == ColumnDistributionType.NUMERICAL
                assert not hasattr(stats.distribution, "histogram")
                assert isinstance(stats.distribution.min, (int, float))
                assert isinstance(stats.distribution.max, (int, float))
                assert isinstance(stats.distribution.mean, float)
                assert isinstance(stats.distribution.stddev, float)
                assert isinstance(stats.distribution.median, float)
