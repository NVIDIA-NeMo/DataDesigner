# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from data_designer.config.analysis.column_statistics import (
    CategoricalDistribution,
    CategoricalHistogramData,
    ColumnDistributionType,
    MissingValue,
    NumericalDistribution,
)
from data_designer.config.column_configs import LLMJudgeColumnConfig, Score
from data_designer.engine.analysis.utils.judge_score_processing import (
    JudgeScoreDistributions,
    JudgeScoreSample,
    extract_judge_score_distributions,
    sample_scores_and_reasoning,
)


def test_extract_judge_score_distributions_numerical_scores(stub_judge_column_config):
    sample_judge_data = [
        {"Quality": {"score": 4, "reasoning": "Excellent implementation"}},
        {"Quality": {"score": 3, "reasoning": "Good implementation"}},
        {"Quality": {"score": 2, "reasoning": "Fair implementation"}},
        {"Quality": {"score": 1, "reasoning": "Poor implementation"}},
        {"Quality": {"score": 0, "reasoning": "Very poor implementation"}},
    ]
    sample_judge_dataframe = pd.DataFrame({"judge_scores": sample_judge_data})
    result = extract_judge_score_distributions(stub_judge_column_config, sample_judge_dataframe)

    assert isinstance(result, JudgeScoreDistributions)
    assert "Quality" in result.scores
    assert result.scores["Quality"] == [4, 3, 2, 1, 0]
    assert result.distribution_types["Quality"] == ColumnDistributionType.NUMERICAL
    assert isinstance(result.distributions["Quality"], NumericalDistribution)
    assert isinstance(result.histograms["Quality"], CategoricalHistogramData)


def test_extract_judge_score_distributions_categorical_scores(stub_judge_column_config):
    mixed_type_judge_data = [
        {"Quality": {"score": 4, "reasoning": "Excellent implementation"}},
        {"Quality": {"score": "good", "reasoning": "Good implementation"}},
        {"Quality": {"score": 2, "reasoning": "Fair implementation"}},
        {"Quality": {"score": "poor", "reasoning": "Poor implementation"}},
    ]
    mixed_type_judge_dataframe = pd.DataFrame({"judge_scores": mixed_type_judge_data})
    result = extract_judge_score_distributions(stub_judge_column_config, mixed_type_judge_dataframe)

    assert isinstance(result, JudgeScoreDistributions)
    assert "Quality" in result.scores
    assert result.scores["Quality"] == [4, "good", 2, "poor"]
    assert result.distribution_types["Quality"] == ColumnDistributionType.CATEGORICAL
    assert isinstance(result.distributions["Quality"], CategoricalDistribution)


def test_extract_judge_score_distributions_edge_cases(stub_judge_column_config):
    malformed_data = [
        {"Quality": {"score": 4, "reasoning": "Good reasoning"}},
        {"Quality": {"reasoning": "Missing score"}},  # Missing score field
        {"Quality": {"score": 2, "reasoning": "Valid entry"}},
    ]
    df = pd.DataFrame({"judge_scores": malformed_data})
    result = extract_judge_score_distributions(stub_judge_column_config, df)
    assert result.scores["Quality"] == [4, "None", 2]

    none_data = [
        {"Quality": {"score": None, "reasoning": "No score provided"}},
        {"Quality": {"score": 4, "reasoning": "Good score"}},
    ]
    df = pd.DataFrame({"judge_scores": none_data})
    result = extract_judge_score_distributions(stub_judge_column_config, df)
    assert result.scores["Quality"] == ["None", 4]
    assert result.distribution_types["Quality"] == ColumnDistributionType.CATEGORICAL

    missing_reasoning_data = [{"Quality": {"score": 4}}]  # Missing reasoning
    df = pd.DataFrame({"judge_scores": missing_reasoning_data})
    result = extract_judge_score_distributions(stub_judge_column_config, df)
    assert result.scores["Quality"] == [4]
    assert result.reasoning["Quality"] == ["No reasoning provided"]

    malformed_data = ["not a dict", {"Quality": "not a dict either"}, {"Quality": {"score": 4, "reasoning": "valid"}}]
    df = pd.DataFrame({"judge_scores": malformed_data})
    result = extract_judge_score_distributions(stub_judge_column_config, df)
    assert result == MissingValue.OUTPUT_FORMAT_ERROR


def test_extract_judge_score_distributions_multiple_scores():
    score1 = Score(
        name="Quality",
        description="Quality assessment",
        options={4: "Excellent", 3: "Good", 2: "Fair", 1: "Poor", 0: "Very Poor"},
    )
    score2 = Score(
        name="Clarity",
        description="Clarity assessment",
        options={4: "Very Clear", 3: "Clear", 2: "Somewhat Clear", 1: "Unclear", 0: "Very Unclear"},
    )

    config = LLMJudgeColumnConfig(
        name="judge_scores", prompt="Evaluate the code", model_alias="test_model", scores=[score1, score2]
    )

    data = [
        {"Quality": {"score": 4, "reasoning": "Excellent quality"}, "Clarity": {"score": 3, "reasoning": "Clear code"}},
        {"Quality": {"score": 2, "reasoning": "Fair quality"}, "Clarity": {"score": 4, "reasoning": "Very clear code"}},
    ]

    df = pd.DataFrame({"judge_scores": data})
    result = extract_judge_score_distributions(config, df)

    assert isinstance(result, JudgeScoreDistributions)
    assert "Quality" in result.scores and "Clarity" in result.scores
    assert result.scores["Quality"] == [4, 2]
    assert result.scores["Clarity"] == [3, 4]


def test_sample_scores_and_reasoning_basic_cases():
    result = sample_scores_and_reasoning([4, 3, 2], ["Good", "Fair", "Poor"], num_samples=5)
    assert len(result) == 3
    assert all(isinstance(sample, JudgeScoreSample) for sample in result)
    assert [sample.score for sample in result] == [4, 3, 2]

    result = sample_scores_and_reasoning([4, 3, 2, 1], ["Excellent", "Good", "Fair", "Poor"], num_samples=4)
    assert len(result) == 4
    assert all(isinstance(sample, JudgeScoreSample) for sample in result)

    scores = [4, 4, 3, 3, 2, 2, 1, 1, 0, 0]
    reasoning = ["Excellent"] * 2 + ["Good"] * 2 + ["Fair"] * 2 + ["Poor"] * 2 + ["Very Poor"] * 2
    result = sample_scores_and_reasoning(scores, reasoning, num_samples=3, random_seed=42)
    assert len(result) == 3
    assert all(isinstance(sample, JudgeScoreSample) for sample in result)


def test_sample_scores_and_reasoning_edge_cases():
    result = sample_scores_and_reasoning(
        [4, 3, 2, 1, 0], ["Excellent", "Good", "Fair", "Poor", "Very Poor"], num_samples=3, random_seed=42
    )
    assert len(result) == 3
    unique_scores = set(sample.score for sample in result)
    assert len(unique_scores) == 3  # One from each category

    result = sample_scores_and_reasoning([4, 4, 4, 4, 4], ["Excellent"] * 5, num_samples=3, random_seed=42)
    assert len(result) == 3
    assert all(sample.score == 4 for sample in result)

    scores = [4, 4, 3, 3, 2, 2, 1, 1, 0, 0]
    reasoning = ["Excellent"] * 2 + ["Good"] * 2 + ["Fair"] * 2 + ["Poor"] * 2 + ["Very Poor"] * 2
    result1 = sample_scores_and_reasoning(scores, reasoning, num_samples=3, random_seed=42)
    result2 = sample_scores_and_reasoning(scores, reasoning, num_samples=3, random_seed=42)
    assert [sample.score for sample in result1] == [sample.score for sample in result2]


def test_sample_scores_and_reasoning_error_cases():
    with pytest.raises(ValueError, match="scores and reasoning must have the same length"):
        sample_scores_and_reasoning([4, 3, 2], ["Good", "Fair"], num_samples=2)

    with pytest.raises(ValueError, match="scores and reasoning must not be empty"):
        sample_scores_and_reasoning([], [], num_samples=5)

    with pytest.raises(ValueError, match="num_samples must be greater than 0"):
        sample_scores_and_reasoning([4, 3, 2], ["Good", "Fair", "Poor"], num_samples=0)
