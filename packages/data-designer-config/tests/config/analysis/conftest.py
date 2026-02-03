# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.config.analysis.column_statistics import (
    ColumnDistributionType,
    LLMTextColumnStatistics,
    SamplerColumnStatistics,
    SeedDatasetColumnStatistics,
)
from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.sampler_params import SamplerType


@pytest.fixture
def sample_seed_dataset_column_stats():
    """Create sample seed dataset column statistics for testing."""
    return SeedDatasetColumnStatistics(
        column_name="seed_column",
        num_records=1000,
        num_null=50,
        num_unique=950,
        pyarrow_dtype="string",
        simple_dtype="str",
        distribution_type=ColumnDistributionType.CATEGORICAL,
        distribution=None,
    )


@pytest.fixture
def sample_llm_text_column_stats():
    """Create sample LLM text column statistics for testing."""
    return LLMTextColumnStatistics(
        column_name="llm_text_column",
        num_records=500,
        num_null=10,
        num_unique=490,
        pyarrow_dtype="string",
        simple_dtype="str",
        output_tokens_mean=150.5,
        output_tokens_median=150.0,
        output_tokens_stddev=25.2,
        input_tokens_mean=50.0,
        input_tokens_median=50.0,
        input_tokens_stddev=10.0,
    )


@pytest.fixture
def sample_sampler_column_stats():
    """Create sample sampler column statistics for testing."""

    return SamplerColumnStatistics(
        column_name="sampler_column",
        num_records=200,
        num_null=0,
        num_unique=200,
        pyarrow_dtype="string",
        simple_dtype="str",
        sampler_type=SamplerType.CATEGORY,
        distribution_type=ColumnDistributionType.CATEGORICAL,
        distribution=None,
    )


@pytest.fixture
def sample_dataset_profiler_results(
    sample_seed_dataset_column_stats, sample_llm_text_column_stats, sample_sampler_column_stats
):
    """Create sample DatasetProfilerResults for testing."""
    return DatasetProfilerResults(
        num_records=1000,
        target_num_records=1000,
        column_statistics=[
            sample_seed_dataset_column_stats,
            sample_llm_text_column_stats,
            sample_sampler_column_stats,
        ],
        side_effect_column_names=["side_effect_1", "side_effect_2"],
        column_profiles=None,
    )
