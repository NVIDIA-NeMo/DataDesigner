# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example: Using allow_resize for 1:N expansion and N:1 retraction."""

from __future__ import annotations

import data_designer.config as dd
from data_designer.interface import DataDesigner
from data_designer.lazy_heavy_imports import pd


@dd.custom_column_generator(required_columns=["topic"], side_effect_columns=["variation_id"])
def expand_to_questions(df: pd.DataFrame) -> pd.DataFrame:
    """Generate 3 questions per topic (1:N expansion)."""
    rows = []
    for _, row in df.iterrows():
        for i in range(3):
            rows.append(
                {
                    "topic": row["topic"],
                    "question": f"Question {i + 1} about {row['topic']}?",
                    "variation_id": i,
                }
            )
    return pd.DataFrame(rows)


@dd.custom_column_generator(required_columns=["topic", "score"])
def filter_high_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only records with score > 0.5 (N:1 retraction)."""
    filtered = df[df["score"] > 0.5].copy()
    filtered["status"] = "passed"
    return filtered


def run_expansion_example() -> None:
    """3 topics -> 9 questions."""
    data_designer = DataDesigner()
    config_builder = dd.DataDesignerConfigBuilder()

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="topic",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["Python", "ML", "Data"]),
        )
    )
    config_builder.add_column(
        dd.CustomColumnConfig(
            name="question",
            generator_function=expand_to_questions,
            generation_strategy=dd.GenerationStrategy.FULL_COLUMN,
            allow_resize=True,
        )
    )

    preview = data_designer.preview(config_builder=config_builder, num_records=3)
    print(f"Expansion: 3 -> {len(preview.dataset)} records")
    print(preview.dataset.to_string())


def run_retraction_example() -> None:
    """10 records -> ~5 (filtered)."""
    data_designer = DataDesigner()
    config_builder = dd.DataDesignerConfigBuilder()

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="topic",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["A", "B", "C", "D", "E"]),
        )
    )
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="score",
            sampler_type=dd.SamplerType.UNIFORM,
            params=dd.UniformSamplerParams(low=0.0, high=1.0),
        )
    )
    config_builder.add_column(
        dd.CustomColumnConfig(
            name="status",
            generator_function=filter_high_scores,
            generation_strategy=dd.GenerationStrategy.FULL_COLUMN,
            allow_resize=True,
        )
    )

    preview = data_designer.preview(config_builder=config_builder, num_records=10)
    print(f"Retraction: 10 -> {len(preview.dataset)} records")
    print(preview.dataset.to_string())


if __name__ == "__main__":
    run_expansion_example()
    # run_retraction_example()
