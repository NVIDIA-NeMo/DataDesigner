# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example: Chaining expand -> retract -> expand resize operations.

Pipeline: 5 topics -> 15 questions (3 per topic) -> ~8 hard questions (filter easy)
          -> ~24 answer variants (3 per question)
"""

from __future__ import annotations

import data_designer.config as dd
from data_designer.interface import DataDesigner
from data_designer.lazy_heavy_imports import pd


# Step 1: Expand — 1:N, generate 3 questions per topic
@dd.custom_column_generator(required_columns=["topic"], side_effect_columns=["question_id", "difficulty"])
def expand_to_questions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for i in range(3):
            rows.append(
                {
                    "topic": row["topic"],
                    "question": f"Q{i + 1} about {row['topic']}?",
                    "question_id": i,
                    "difficulty": ["easy", "medium", "hard"][i],
                }
            )
    return pd.DataFrame(rows)


# Step 2: Retract — N:1, keep only medium/hard questions
@dd.custom_column_generator(required_columns=["difficulty"])
def filter_non_easy(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["difficulty"] != "easy"].copy().assign(filtered=True)


# Step 3: Expand again — 1:N, generate 3 answer variants per surviving question
@dd.custom_column_generator(required_columns=["question"], side_effect_columns=["variant"])
def expand_to_answers(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for v in range(3):
            rows.append({**row.to_dict(), "answer": f"Answer v{v} to: {row['question']}", "variant": v})
    return pd.DataFrame(rows)


def main() -> None:
    data_designer = DataDesigner()
    config_builder = dd.DataDesignerConfigBuilder()

    # Seed: 5 topics
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="topic",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["Python", "ML", "Data", "Stats", "SQL"]),
        )
    )

    # Expand: 5 topics -> 15 questions
    config_builder.add_column(
        dd.CustomColumnConfig(
            name="question",
            generator_function=expand_to_questions,
            generation_strategy=dd.GenerationStrategy.FULL_COLUMN,
            allow_resize=True,
        )
    )

    # Retract: 15 -> 10 (drop "easy" questions)
    config_builder.add_column(
        dd.CustomColumnConfig(
            name="filtered",
            generator_function=filter_non_easy,
            generation_strategy=dd.GenerationStrategy.FULL_COLUMN,
            allow_resize=True,
        )
    )

    # Expand again: 10 -> 30 answer variants
    config_builder.add_column(
        dd.CustomColumnConfig(
            name="answer",
            generator_function=expand_to_answers,
            generation_strategy=dd.GenerationStrategy.FULL_COLUMN,
            allow_resize=True,
        )
    )

    # Preview (single batch)
    preview = data_designer.preview(config_builder=config_builder, num_records=5)
    print(f"Preview: 5 topics -> {len(preview.dataset)} answer variants")
    print(preview.dataset[["topic", "difficulty", "question", "variant", "answer"]].to_string())
    print()

    # Build (multiple batches: 10 records with buffer_size=3 -> 4 batches)
    data_designer.set_run_config(dd.RunConfig(buffer_size=3))
    results = data_designer.create(config_builder=config_builder, num_records=10)
    df = results.load_dataset()
    print(f"Build: 10 topics (4 batches of 3+3+3+1) -> {len(df)} answer variants")
    print(df[["topic", "difficulty", "question", "variant"]].to_string())


if __name__ == "__main__":
    main()
