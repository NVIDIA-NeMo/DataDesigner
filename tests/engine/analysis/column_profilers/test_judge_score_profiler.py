# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pandas as pd
import pyarrow as pa
import pytest

from data_designer.config.analysis.column_statistics import (
    CategoricalDistribution,
    CategoricalHistogramData,
    ColumnDistributionType,
    MissingValue,
    NumericalDistribution,
)
from data_designer.config.column_configs import LLMJudgeColumnConfig, Score
from data_designer.engine.analysis.column_profilers.base import ColumnConfigWithDataFrame
from data_designer.engine.analysis.column_profilers.judge_score_profiler import (
    JudgeScoreProfiler,
    JudgeScoreProfilerConfig,
    JudgeScoreProfilerResults,
    JudgeScoreSummary,
)
from data_designer.engine.analysis.utils.judge_score_processing import JudgeScoreDistributions, JudgeScoreSample


@pytest.fixture
def stub_judge_data():
    return [
        {"Quality": {"score": 4, "reasoning": "Excellent implementation"}},
        {"Quality": {"score": 3, "reasoning": "Good implementation"}},
        {"Quality": {"score": 2, "reasoning": "Fair implementation"}},
        {"Quality": {"score": 1, "reasoning": "Poor implementation"}},
        {"Quality": {"score": 0, "reasoning": "Very poor implementation"}},
    ]


@pytest.fixture
def stub_dataframe(stub_judge_data):
    df = pd.DataFrame({"judge_scores": stub_judge_data})
    # Convert to PyArrow backend as required by ColumnConfigWithDataFrame
    return pa.Table.from_pandas(df).to_pandas(types_mapper=pd.ArrowDtype)


@pytest.fixture
def stub_extract_judge_score_distributions():
    """Fixture for mocking extract_judge_score_distributions function."""
    with patch(
        "data_designer.engine.analysis.column_profilers.judge_score_profiler.extract_judge_score_distributions",
        autospec=True,
    ) as mock_extract:
        yield mock_extract


@pytest.fixture
def stub_sample_scores_and_reasoning():
    """Fixture for mocking sample_scores_and_reasoning function."""
    with patch(
        "data_designer.engine.analysis.column_profilers.judge_score_profiler.sample_scores_and_reasoning",
        autospec=True,
    ) as mock_sample:
        yield mock_sample


@pytest.fixture
def stub_judge_score_profiler(stub_resource_provider):
    config = JudgeScoreProfilerConfig(model_alias="test_model", summary_score_sample_size=5)
    return JudgeScoreProfiler(config=config, resource_provider=stub_resource_provider)


def test_judge_score_profiler_profile_success(
    stub_sample_scores_and_reasoning,
    stub_extract_judge_score_distributions,
    stub_judge_score_profiler,
    stub_judge_column_config,
    stub_dataframe,
    stub_judge_distributions,
):
    stub_extract_judge_score_distributions.return_value = stub_judge_distributions
    mock_samples = [JudgeScoreSample(score=4, reasoning="Excellent"), JudgeScoreSample(score=3, reasoning="Good")]
    stub_sample_scores_and_reasoning.return_value = mock_samples

    params = ColumnConfigWithDataFrame(column_config=stub_judge_column_config, df=stub_dataframe)

    result = stub_judge_score_profiler.profile(params)

    assert isinstance(result, JudgeScoreProfilerResults)
    assert len(result.summaries) == 1
    assert result.summaries["Quality"].score_name == "Quality"
    assert result.score_distributions == stub_judge_distributions

    stub_extract_judge_score_distributions.assert_called_once_with(stub_judge_column_config, stub_dataframe)
    stub_sample_scores_and_reasoning.assert_called_once()


def test_judge_score_profiler_profile_with_none_sample_size(
    stub_extract_judge_score_distributions,
    stub_judge_column_config,
    stub_dataframe,
    stub_resource_provider,
    stub_judge_distributions,
):
    config = JudgeScoreProfilerConfig(model_alias="test_model", summary_score_sample_size=None)
    profiler = JudgeScoreProfiler(config=config, resource_provider=stub_resource_provider)

    stub_extract_judge_score_distributions.return_value = stub_judge_distributions
    params = ColumnConfigWithDataFrame(column_config=stub_judge_column_config, df=stub_dataframe)

    result = profiler.profile(params)

    assert isinstance(result, JudgeScoreProfilerResults)
    assert result.summaries == {}
    assert result.score_distributions == stub_judge_distributions


def test_judge_score_profiler_multiple_rubrics(
    stub_sample_scores_and_reasoning,
    stub_extract_judge_score_distributions,
    stub_judge_score_profiler,
    stub_dataframe,
):
    score1 = Score(name="Quality", description="Quality assessment", options={4: "Excellent", 3: "Good"})
    score2 = Score(name="Clarity", description="Clarity assessment", options={4: "Clear", 3: "Unclear"})
    config = LLMJudgeColumnConfig(
        name="judge_scores", prompt="Evaluate", model_alias="test_model", scores=[score1, score2]
    )

    mock_distributions = JudgeScoreDistributions(
        scores={"Quality": [4, 3], "Clarity": [4, 3]},
        reasoning={"Quality": ["Excellent", "Good"], "Clarity": ["Clear", "Unclear"]},
        distribution_types={"Quality": ColumnDistributionType.NUMERICAL, "Clarity": ColumnDistributionType.NUMERICAL},
        distributions={
            "Quality": Mock(spec=NumericalDistribution, mean=3.5, stddev=0.5, min=3, max=4, median=3.5),
            "Clarity": Mock(spec=NumericalDistribution, mean=3.0, stddev=0.0, min=3, max=3, median=3.0),
        },
        histograms={
            "Quality": Mock(spec=CategoricalHistogramData, categories=[4, 3], counts=[10, 8]),
            "Clarity": Mock(spec=CategoricalHistogramData, categories=[4, 3], counts=[12, 6]),
        },
    )
    stub_extract_judge_score_distributions.return_value = mock_distributions
    stub_sample_scores_and_reasoning.return_value = [JudgeScoreSample(score=4, reasoning="Test")]

    params = ColumnConfigWithDataFrame(column_config=config, df=stub_dataframe)

    result = stub_judge_score_profiler.profile(params)

    assert len(result.summaries) == 2
    assert stub_sample_scores_and_reasoning.call_count == 2


def test_judge_score_profiler_integration_workflow(
    stub_sample_scores_and_reasoning,
    stub_extract_judge_score_distributions,
    stub_judge_column_config,
    stub_dataframe,
    stub_resource_provider,
):
    mock_distributions = JudgeScoreDistributions(
        scores={"Quality": [4, 3, 2, 1, 0]},
        reasoning={"Quality": ["Excellent", "Good", "Fair", "Poor", "Very Poor"]},
        distribution_types={"Quality": ColumnDistributionType.NUMERICAL},
        distributions={"Quality": NumericalDistribution(min=0, max=4, mean=2.0, stddev=1.4, median=2.0)},
        histograms={"Quality": CategoricalHistogramData(categories=[4, 3, 2, 1, 0], counts=[1, 1, 1, 1, 1])},
    )
    stub_extract_judge_score_distributions.return_value = mock_distributions
    stub_sample_scores_and_reasoning.return_value = [
        JudgeScoreSample(score=4, reasoning="Excellent"),
        JudgeScoreSample(score=2, reasoning="Fair"),
    ]

    config = JudgeScoreProfilerConfig(model_alias="test_model", summary_score_sample_size=2)
    profiler = JudgeScoreProfiler(config=config, resource_provider=stub_resource_provider)
    params = ColumnConfigWithDataFrame(column_config=stub_judge_column_config, df=stub_dataframe)

    result = profiler.profile(params)

    assert isinstance(result, JudgeScoreProfilerResults)
    assert len(result.summaries) == 1
    assert result.summaries["Quality"].score_name == "Quality"
    assert result.score_distributions == mock_distributions
    stub_extract_judge_score_distributions.assert_called_once()
    stub_sample_scores_and_reasoning.assert_called_once()


def test_judge_score_profiler_summarize_score_sample(stub_judge_score_profiler, stub_model_facade):
    sample = [JudgeScoreSample(score=4, reasoning="Excellent"), JudgeScoreSample(score=3, reasoning="Good")]
    histogram = CategoricalHistogramData(categories=[4, 3, 2, 1, 0], counts=[10, 8, 5, 2, 1])
    distribution = NumericalDistribution(min=0, max=4, mean=2.5, stddev=1.2, median=2.0)

    result = stub_judge_score_profiler._summarize_score_sample(
        name="quality",
        sample=sample,
        histogram=histogram,
        distribution=distribution,
        distribution_type=ColumnDistributionType.NUMERICAL,
    )

    assert isinstance(result, JudgeScoreSummary)
    assert result.score_name == "quality"
    assert result.summary == "Generated summary text"
    assert result.score_samples == sample
    stub_model_facade.generate.assert_called_once()


def test_judge_score_profiler_summarize_score_edge_cases(stub_judge_score_profiler):
    sample = [JudgeScoreSample(score=4, reasoning="Test")]

    result = stub_judge_score_profiler._summarize_score_sample(
        name="quality",
        sample=sample,
        histogram=Mock(spec=CategoricalHistogramData),
        distribution=MissingValue.CALCULATION_FAILED,
        distribution_type=ColumnDistributionType.UNKNOWN,
    )
    assert result.summary == "No judge score information available to summarize."

    result = stub_judge_score_profiler._summarize_score_sample(
        name="quality",
        sample=[],
        histogram=Mock(spec=CategoricalHistogramData),
        distribution=Mock(spec=NumericalDistribution),
        distribution_type=ColumnDistributionType.NUMERICAL,
    )
    assert result.summary == "No judge score information available to summarize."


def test_judge_score_profiler_summarize_score_categorical_distribution(stub_judge_score_profiler, stub_model_facade):
    sample = [JudgeScoreSample(score="good", reasoning="Good implementation")]
    histogram = CategoricalHistogramData(categories=["excellent", "good", "fair", "poor"], counts=[5, 10, 3, 2])
    distribution = CategoricalDistribution(most_common_value="good", least_common_value="poor", histogram=histogram)

    result = stub_judge_score_profiler._summarize_score_sample(
        name="quality",
        sample=sample,
        histogram=histogram,
        distribution=distribution,
        distribution_type=ColumnDistributionType.CATEGORICAL,
    )

    assert result.score_name == "quality"
    assert result.summary == "Generated summary text"
    stub_model_facade.generate.assert_called_once()


def test_judge_score_profiler_summarize_score_model_failure(stub_judge_score_profiler, stub_model_facade):
    stub_model_facade.generate.side_effect = Exception("Model generation failed")
    sample = [JudgeScoreSample(score=4, reasoning="Test")]
    histogram = CategoricalHistogramData(categories=[4], counts=[1])
    distribution = NumericalDistribution(min=4, max=4, mean=4.0, stddev=0.0, median=4.0)

    result = stub_judge_score_profiler._summarize_score_sample(
        name="quality",
        sample=sample,
        histogram=histogram,
        distribution=distribution,
        distribution_type=ColumnDistributionType.NUMERICAL,
    )

    assert result.summary == "Score summarization failed: Model generation failed"


def test_judge_score_profiler_prompt_construction(stub_judge_score_profiler, stub_model_facade):
    sample = [JudgeScoreSample(score=4, reasoning="Excellent"), JudgeScoreSample(score=2, reasoning="Fair")]
    histogram = CategoricalHistogramData(categories=[4, 3, 2, 1, 0], counts=[10, 8, 5, 2, 1])
    distribution = NumericalDistribution(min=0, max=4, mean=2.5, stddev=1.2, median=2.0)

    stub_judge_score_profiler._summarize_score_sample(
        name="quality",
        sample=sample,
        histogram=histogram,
        distribution=distribution,
        distribution_type=ColumnDistributionType.NUMERICAL,
    )

    call_args = stub_model_facade.generate.call_args
    prompt = call_args[1]["prompt"]
    system_prompt = call_args[1]["system_prompt"]

    assert "quality" in prompt
    assert "Excellent" in prompt
    assert "Mean score: 2.50" in prompt
    assert "YOU WILL PRODUCE LESS THAN 75 WORDS" in prompt
    assert "expert at distilling complex feedback" in system_prompt
    assert "Focus on specificity and balance" in system_prompt
