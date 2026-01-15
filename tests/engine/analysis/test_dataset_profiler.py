# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.column_configs import SamplerColumnConfig
from data_designer.config.sampler_params import CategorySamplerParams, SamplerType
from data_designer.engine.analysis.column_profilers.judge_score_profiler import JudgeScoreProfilerConfig
from data_designer.engine.analysis.dataset_profiler import DataDesignerDatasetProfiler, DatasetProfilerConfig
from data_designer.engine.analysis.errors import DatasetProfilerConfigurationError
from data_designer.engine.analysis.utils.judge_score_processing import JudgeScoreSample
from data_designer.engine.dataset_builders.multi_column_configs import SamplerMultiColumnConfig


def test_dataset_profiler_config_flattens_multi_column_configs():
    multi_config = SamplerMultiColumnConfig(
        columns=[
            SamplerColumnConfig(
                name="col1",
                sampler_type=SamplerType.CATEGORY,
                params=CategorySamplerParams(values=["a", "b", "c"]),
            ),
            SamplerColumnConfig(
                name="col2",
                sampler_type=SamplerType.CATEGORY,
                params=CategorySamplerParams(values=["d", "e", "f"]),
            ),
        ]
    )

    config = DatasetProfilerConfig(column_configs=[multi_config])

    assert len(config.column_configs) == 2
    assert config.column_configs[0].name == "col1"
    assert config.column_configs[1].name == "col2"


def test_dataset_profiler_config_raises_error_when_all_columns_dropped():
    column_configs = [
        SamplerColumnConfig(
            name="test_id",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["a", "b", "c"]),
            drop=True,
        ),
    ]

    with pytest.raises(DatasetProfilerConfigurationError, match="All columns were dropped!"):
        DatasetProfilerConfig(column_configs=column_configs)


@patch(
    "data_designer.engine.analysis.column_profilers.judge_score_profiler.extract_judge_score_distributions",
    autospec=True,
)
@patch(
    "data_designer.engine.analysis.column_profilers.judge_score_profiler.sample_scores_and_reasoning",
    autospec=True,
)
def test_dataset_profiler_profile_dataset_with_column_profilers(
    mock_sample_scores,
    mock_extract_distributions,
    dataset_profiler,
    stub_df,
    stub_judge_distributions,
    stub_model_facade,
):
    mock_extract_distributions.return_value = stub_judge_distributions
    mock_sample_scores.return_value = [
        JudgeScoreSample(score=4, reasoning="Excellent implementation"),
        JudgeScoreSample(score=2, reasoning="Fair implementation"),
    ]

    profiler_config = JudgeScoreProfilerConfig(model_alias="nano", summary_score_sample_size=5)
    dataset_profiler.config.column_profiler_configs = [profiler_config]
    dataset_profiler.resource_provider.model_registry.get_model.return_value = stub_model_facade

    profile = dataset_profiler.profile_dataset(len(stub_df), stub_df)

    assert isinstance(profile, DatasetProfilerResults)
    assert len(profile.column_profiles) == 1

    mock_extract_distributions.assert_called()
    mock_sample_scores.assert_called()
    stub_model_facade.generate.assert_called()


@patch(
    "data_designer.engine.analysis.dataset_profiler.DataDesignerDatasetProfiler._validate_schema_consistency",
    autospec=True,
)
def test_dataset_profiler_requires_model_registry_with_column_profiler_configs(
    mock_validate_schema_consistency, stub_resource_provider_no_model_registry
):
    column_configs = [
        SamplerColumnConfig(
            name="test_id",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["a", "b", "c"]),
        ),
    ]

    mock_validate_schema_consistency.return_value = None

    DataDesignerDatasetProfiler(
        config=DatasetProfilerConfig(
            column_configs=column_configs,
        ),
        resource_provider=stub_resource_provider_no_model_registry,
    )

    with pytest.raises(
        DatasetProfilerConfigurationError,
        match="Model registry is required for column profiler configs",
    ):
        DataDesignerDatasetProfiler(
            config=DatasetProfilerConfig(
                column_configs=column_configs,
                column_profiler_configs=[
                    JudgeScoreProfilerConfig(
                        model_alias="model-alias",
                        summary_score_sample_size=5,
                    )
                ],
            ),
            resource_provider=stub_resource_provider_no_model_registry,
        )
