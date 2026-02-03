# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# %% [markdown]
# # Custom Column Examples
#
# This notebook demonstrates custom column generators with the decorator-based API:
# - Basic transforms (1-arg, 2-arg, 3-arg signatures)
# - Multi-turn LLM workflows
# - Benchmarking cell_by_cell vs full_column strategies

# %% Imports
from __future__ import annotations

import time

import pandas as pd
from pydantic import BaseModel

import data_designer.config as dd
from data_designer.interface import DataDesigner

MODEL_ALIAS = "nvidia-text"

# %% [markdown]
# ## Basic Examples: Function Signatures


# %% 1-arg: Simple transform (no params, no LLM)
@dd.custom_column_generator(required_columns=["name"])
def simple_greeting(row: dict) -> dict:
    row["greeting"] = f"Hello, {row['name']}!"
    return row


# %% 2-arg: With typed params
class MessageParams(BaseModel):
    tone: str = "friendly"
    max_words: int = 50


@dd.custom_column_generator(required_columns=["name", "product_interest"])
def greeting_with_params(row: dict, generator_params: MessageParams) -> dict:
    row["greeting"] = f"Hello, {row['name']}! Interested in {row['product_interest']}. (tone: {generator_params.tone})"
    return row


# %% 3-arg: With generator_params and LLM access
@dd.custom_column_generator(
    required_columns=["name", "product_interest"],
    side_effect_columns=["prompt"],
    model_aliases=[MODEL_ALIAS],
)
def generate_personalized_message(row: dict, generator_params: MessageParams, models: dict) -> dict:
    prompt = f"Write a {generator_params.tone} message for {row['name']} about {row['product_interest']}. Max {generator_params.max_words} words."
    response, _ = models[MODEL_ALIAS].generate(prompt=prompt)
    row["personalized_message"] = response
    row["prompt"] = prompt
    return row


# %% Run basic pipeline
data_designer = DataDesigner()
config = dd.DataDesignerConfigBuilder()

config.add_column(
    dd.SamplerColumnConfig(
        name="name",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=["Alice", "Bob", "Charlie"]),
    )
)
config.add_column(
    dd.SamplerColumnConfig(
        name="product_interest",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=["Electronics", "Books", "Sports"]),
    )
)
config.add_column(dd.CustomColumnConfig(name="greeting", generator_function=simple_greeting))
config.add_column(
    dd.CustomColumnConfig(
        name="personalized_message",
        generator_function=generate_personalized_message,
        generator_params=MessageParams(tone="professional", max_words=30),
    )
)

preview = data_designer.preview(config_builder=config, num_records=3)
preview.display_sample_record()

# %% [markdown]
# ## Multi-turn LLM Workflow
#
# Writer drafts → Editor critiques → Writer revises


# %% Multi-turn generator
@dd.custom_column_generator(
    required_columns=["topic"],
    side_effect_columns=["conversation_trace"],
    model_aliases=[MODEL_ALIAS],
)
def writer_editor_workflow(row: dict, generator_params: None, models: dict) -> dict:
    topic = row["topic"]
    model = models[MODEL_ALIAS]

    draft, _ = model.generate(prompt=f"Write a 2-sentence hook about '{topic}'.")
    critique, _ = model.generate(prompt=f"Give one improvement for: {draft}")
    revised, _ = model.generate(prompt=f"Revise based on feedback:\n\nOriginal: {draft}\n\nFeedback: {critique}")

    row["refined_intro"] = revised
    row["conversation_trace"] = f"Draft: {draft[:50]}... | Critique: {critique[:50]}..."
    return row


# %% Run multi-turn pipeline
config_multiturn = dd.DataDesignerConfigBuilder()
config_multiturn.add_column(
    dd.SamplerColumnConfig(
        name="topic",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=["sustainable living", "remote work", "learning languages"]),
    )
)
config_multiturn.add_column(dd.CustomColumnConfig(name="refined_intro", generator_function=writer_editor_workflow))

preview = data_designer.preview(config_builder=config_multiturn, num_records=2)
preview.display_sample_record()

# %% [markdown]
# ## Experimentation
#
# Test custom column functions with real LLM calls without running the full pipeline.

# %% Test a generator function directly
models = data_designer.get_models([MODEL_ALIAS])

# Call the function directly with a sample row
sample_row = {"topic": "quantum computing"}
result = writer_editor_workflow(sample_row, None, models)

print(f"Input: {sample_row['topic']}")
print(f"Output: {result['refined_intro'][:100]}...")

# %% [markdown]
# ## Generation Strategies
#
# - `cell_by_cell` (default): Framework handles parallelization. Recommended for LLM calls.
# - `full_column`: For vectorized DataFrame operations. Not recommended for LLM calls.


# %% Define cell-by-cell strategy (recommended for LLM calls)
@dd.custom_column_generator(required_columns=["topic"], model_aliases=[MODEL_ALIAS])
def cell_by_cell_generator(row: dict, generator_params: None, models: dict) -> dict:
    """Row-by-row processing. Framework handles parallelization."""
    response, _ = models[MODEL_ALIAS].generate(prompt=f"Write a one-sentence fact about: {row['topic']}")
    row["fact"] = response
    return row


# %% Full-column strategy (for vectorized ops, not recommended for LLM calls)
@dd.custom_column_generator(required_columns=["topic"])
def full_column_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Batch processing for vectorized operations (no LLM)."""
    df["topic_upper"] = df["topic"].str.upper()
    return df


# %% Run examples
NUM_RECORDS = 6

topic_config = dd.SamplerColumnConfig(
    name="topic",
    sampler_type=dd.SamplerType.CATEGORY,
    params=dd.CategorySamplerParams(values=["space", "history", "biology", "physics"]),
)

# cell_by_cell with LLM (recommended)
config_cell = dd.DataDesignerConfigBuilder()
config_cell.add_column(topic_config)
config_cell.add_column(dd.CustomColumnConfig(name="fact", generator_function=cell_by_cell_generator))

start = time.perf_counter()
result_cell = data_designer.preview(config_builder=config_cell, num_records=NUM_RECORDS)
elapsed_cell = time.perf_counter() - start

print(f"cell_by_cell with LLM: {elapsed_cell:.2f}s")
result_cell.display_sample_record()

# full_column for vectorized ops (no LLM)
config_full = dd.DataDesignerConfigBuilder()
config_full.add_column(topic_config)
config_full.add_column(
    dd.CustomColumnConfig(
        name="topic_upper",
        generator_function=full_column_transform,
        generation_strategy=dd.GenerationStrategy.FULL_COLUMN,
    )
)

result_full = data_designer.preview(config_builder=config_full, num_records=NUM_RECORDS)
print(result_full.dataset[["topic", "topic_upper"]].head())
