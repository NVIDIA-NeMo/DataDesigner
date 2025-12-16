# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from typing import Any, Optional, Union

import pandas as pd

from data_designer.config.analysis.column_profilers import JudgeScoreDistributions, JudgeScoreSample
from data_designer.config.analysis.column_statistics import (
    CategoricalDistribution,
    ColumnDistributionType,
    MissingValue,
    NumericalDistribution,
)
from data_designer.config.column_configs import LLMJudgeColumnConfig

logger = logging.getLogger(__name__)


def extract_judge_score_distributions(
    column_config: LLMJudgeColumnConfig, df: pd.DataFrame
) -> Union[JudgeScoreDistributions, MissingValue]:
    scores = defaultdict(list)
    reasoning = defaultdict(list)

    # Aggregate results as dicts of form {score_name: <result>}.
    histograms = {}
    distributions = {}
    distribution_types = {}

    for score in column_config.scores:
        is_numerical = True
        name = score.name
        for results in df[column_config.name]:
            try:
                score = results[name].get("score", None)

                if _can_be_converted_to_int(score):
                    score = int(score)
                else:
                    score = str(score)
                    is_numerical = False

                scores[name].append(score)
                reasoning[name].append(results[name].get("reasoning", "No reasoning provided"))
            except Exception as e:
                logger.warning(f"⚠️ Failed to extract judge score for '{name}': {e}")
                return MissingValue.OUTPUT_FORMAT_ERROR

        try:
            series = pd.Series(scores[name], name=name)
            cat_dist = CategoricalDistribution.from_series(series)

            # For judge scores, build a categorical histogram, since numerical scores are integers.
            histograms[name] = cat_dist.histogram

            if is_numerical:
                distribution_types[name] = ColumnDistributionType.NUMERICAL
                distributions[name] = NumericalDistribution.from_series(series)
            else:
                distribution_types[name] = ColumnDistributionType.CATEGORICAL
                distributions[name] = cat_dist

        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate judge score distribution for '{name}': {e}")
            distribution_types[name] = ColumnDistributionType.UNKNOWN
            distributions[name] = MissingValue.CALCULATION_FAILED
            histograms[name] = MissingValue.CALCULATION_FAILED

    return JudgeScoreDistributions(
        scores=dict(scores),
        reasoning=dict(reasoning),
        distribution_types=distribution_types,
        distributions=distributions,
        histograms=histograms,
    )


def sample_scores_and_reasoning(
    scores: list[Union[int, str]],
    reasoning: list[str],
    num_samples: int,
    random_seed: Optional[int] = None,
) -> list[JudgeScoreSample]:
    if len(scores) != len(reasoning):
        raise ValueError("scores and reasoning must have the same length")

    if len(scores) == 0:
        raise ValueError("scores and reasoning must not be empty")

    if num_samples <= 0:
        raise ValueError("num_samples must be greater than 0")

    df_samples = pd.DataFrame({"score": scores, "reasoning": reasoning})

    if len(scores) <= num_samples:
        return [JudgeScoreSample(score=score, reasoning=reasoning) for score, reasoning in zip(scores, reasoning)]

    # Sample maintaining original proportions from each category (int or str)
    # Calculate the frequency of each score category
    score_category_counts = df_samples["score"].value_counts()

    # If more categories than samples, pick one sample from each of the most frequent categories
    if len(score_category_counts) >= num_samples:
        top_categories = score_category_counts.head(num_samples).index
        samples = pd.concat(
            [df_samples[df_samples["score"] == cat].sample(n=1, random_state=random_seed) for cat in top_categories],
            ignore_index=True,
        )
    else:
        # Sample proportionally to maintain original category ratios
        # Create weights based on the original frequency of each score
        weights = df_samples["score"].map(score_category_counts)
        samples = df_samples.sample(n=num_samples, weights=weights, random_state=random_seed)

    return [
        JudgeScoreSample(score=row["score"], reasoning=row["reasoning"]) for row in samples.to_dict(orient="records")
    ]


def _can_be_converted_to_int(value: Any) -> bool:
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False
