# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from data_designer.config.analysis.column_profilers import (
    JudgeScoreProfilerConfig,
    JudgeScoreProfilerResults,
    JudgeScoreSummary,
)
from data_designer.config.analysis.column_statistics import (
    ColumnDistributionType,
    MissingValue,
)
from data_designer.config.column_types import DataDesignerColumnType
from data_designer.engine.analysis.column_profilers.base import ColumnConfigWithDataFrame, ColumnProfiler
from data_designer.engine.analysis.utils.judge_score_processing import (
    extract_judge_score_distributions,
    sample_scores_and_reasoning,
)
from data_designer.engine.models.recipes.response_recipes import TextResponseRecipe

if TYPE_CHECKING:
    from data_designer.config.analysis.column_profilers import JudgeScoreSample
    from data_designer.config.analysis.column_statistics import (
        CategoricalDistribution,
        CategoricalHistogramData,
        NumericalDistribution,
    )
    from data_designer.engine.models.facade import ModelFacade

logger = logging.getLogger(__name__)


class JudgeScoreProfiler(ColumnProfiler[JudgeScoreProfilerConfig]):
    @staticmethod
    def get_applicable_column_types() -> list[DataDesignerColumnType]:
        return [DataDesignerColumnType.LLM_JUDGE]

    def get_model(self, model_alias: str) -> ModelFacade:
        return self.resource_provider.model_registry.get_model(model_alias=model_alias)

    def profile(self, column_config_with_df: ColumnConfigWithDataFrame) -> JudgeScoreProfilerResults:
        column_config, df = column_config_with_df.as_tuple()

        logger.info(
            f"{column_config.get_column_emoji()} Analyzing LLM-as-judge scores for column: '{column_config.name}'"
        )

        score_summaries = {}
        score_distributions = extract_judge_score_distributions(column_config, df)

        if self.config.summary_score_sample_size is None or isinstance(score_distributions, MissingValue):
            return JudgeScoreProfilerResults(
                summaries={},
                column_name=column_config.name,
                score_distributions=score_distributions,
            )

        for score in column_config.scores:
            score_name = score.name
            logger.info(f"{random.choice(['👩‍⚖️', '👨‍⚖️'])} Summarizing LLM-as-judge score: '{score_name}'")
            score_sample = sample_scores_and_reasoning(
                scores=score_distributions.scores[score_name],
                reasoning=score_distributions.reasoning[score_name],
                num_samples=self.config.summary_score_sample_size,
            )

            score_summaries[score_name] = self._summarize_score_sample(
                name=score_name,
                sample=score_sample,
                histogram=score_distributions.histograms[score_name],
                distribution=score_distributions.distributions[score_name],
                distribution_type=score_distributions.distribution_types[score_name],
            )

        return JudgeScoreProfilerResults(
            column_name=column_config.name,
            summaries=score_summaries,
            score_distributions=score_distributions,
        )

    def _summarize_score_sample(
        self,
        name: str,
        sample: list[JudgeScoreSample],
        histogram: CategoricalHistogramData,
        distribution: CategoricalDistribution | NumericalDistribution | MissingValue,
        distribution_type: ColumnDistributionType,
    ) -> JudgeScoreSummary:
        if isinstance(distribution, MissingValue) or not sample:
            return JudgeScoreSummary(
                score_name=name,
                summary="No judge score information available to summarize.",
                score_samples=sample,
            )

        category_info = []
        total_count = sum(histogram.counts)
        for cat, count in zip(histogram.categories, histogram.counts):
            percentage = (count / total_count) * 100
            category_info.append(f"{cat}: {count} records ({percentage:.1f}%)")

        distribution_context = f"Score distribution - {', '.join(category_info)}, "
        if distribution_type == ColumnDistributionType.CATEGORICAL:
            distribution_context += f"Most common value: {distribution.most_common_value}. "
        if distribution_type == ColumnDistributionType.NUMERICAL:
            distribution_context += f"Mean score: {distribution.mean:.2f}. "

        logger.info(f"  |-- number of score samples: {len(sample)}")
        logger.info(f"  |-- {distribution_context.lower()}")

        combined_reasoning = "\n".join([r.reasoning for r in sample])
        prompt = (
            f"Based on the following evaluator reasoning for the '{name}' criterion, "
            "provide a concise summary that captures both the strengths and areas for improvement mentioned. "
            "Be specific about what worked well and what needs improvement.\n\n"
            f"Overall distribution of scores: {distribution_context}"
            f"\nA sample of reasoning:\n{combined_reasoning}\n\n"
            "Do not include any titles like `Summary` or `Summary:`. "
            "Do not wrap the summary in quotation marks. "
            "YOU WILL PRODUCE LESS THAN 75 WORDS in a readable sentence format. "
            "No need to use bullets or headers. Write naturally."
        )

        system_prompt = (
            "You are an expert at distilling complex feedback into concise summaries. "
            "Focus on specificity and balance, incorporating both the distribution context and individual reasoning examples."
        )

        try:
            model = self.get_model(self.config.model_alias)
            recipe = TextResponseRecipe()
            summary, _ = model.generate(
                prompt=recipe.apply_recipe_to_user_prompt(prompt),
                system_prompt=recipe.apply_recipe_to_system_prompt(system_prompt),
                parser=recipe.parse,
            )
            return JudgeScoreSummary(
                score_name=name,
                summary=summary.strip(),
                score_samples=sample,
            )
        except Exception as e:
            return JudgeScoreSummary(
                score_name=name,
                summary=f"Score summarization failed: {e}",
                score_samples=sample,
            )
