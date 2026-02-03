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
def greeting_with_params(row: dict, params: MessageParams) -> dict:
    row["greeting"] = f"Hello, {row['name']}! Interested in {row['product_interest']}. (tone: {params.tone})"
    return row


# %% 3-arg: With params and LLM access
@dd.custom_column_generator(
    required_columns=["name", "product_interest"],
    side_effect_columns=["prompt"],
    model_aliases=[MODEL_ALIAS],
)
def generate_personalized_message(row: dict, params: MessageParams, ctx: dd.CustomColumnContext) -> dict:
    prompt = f"Write a {params.tone} message for {row['name']} about {row['product_interest']}. Max {params.max_words} words."
    row[ctx.column_name] = ctx.generate_text(model_alias=MODEL_ALIAS, prompt=prompt)
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
def writer_editor_workflow(row: dict, params: None, ctx: dd.CustomColumnContext) -> dict:
    topic = row["topic"]

    draft = ctx.generate_text(model_alias=MODEL_ALIAS, prompt=f"Write a 2-sentence hook about '{topic}'.")
    critique = ctx.generate_text(model_alias=MODEL_ALIAS, prompt=f"Give one improvement for: {draft}")
    revised = ctx.generate_text(
        model_alias=MODEL_ALIAS, prompt=f"Revise based on feedback:\n\nOriginal: {draft}\n\nFeedback: {critique}"
    )

    row[ctx.column_name] = revised
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
# ## Benchmark: cell_by_cell vs full_column
#
# Compare row-by-row (framework parallelizes) vs batch processing (user controls parallelism).


# %% Define both strategies
@dd.custom_column_generator(required_columns=["topic"], model_aliases=[MODEL_ALIAS])
def cell_by_cell_generator(row: dict, params: None, ctx: dd.CustomColumnContext) -> dict:
    """Row-by-row, framework parallelizes."""
    row[ctx.column_name] = ctx.generate_text(
        model_alias=MODEL_ALIAS, prompt=f"Write a one-sentence fact about: {row['topic']}"
    )
    return row


@dd.custom_column_generator(required_columns=["topic"], model_aliases=[MODEL_ALIAS])
def full_column_generator(df: pd.DataFrame, params: None, ctx: dd.CustomColumnContext) -> pd.DataFrame:
    """Batch processing with generate_text_batch."""
    prompts = [f"Write a one-sentence fact about: {topic}" for topic in df["topic"]]
    df["fact"] = ctx.generate_text_batch(model_alias=MODEL_ALIAS, prompts=prompts, max_workers=8)
    return df


# %% Run benchmark
NUM_RECORDS = 6

topic_config = dd.SamplerColumnConfig(
    name="topic",
    sampler_type=dd.SamplerType.CATEGORY,
    params=dd.CategorySamplerParams(values=["space", "history", "biology", "physics"]),
)

# cell_by_cell
config_cell = dd.DataDesignerConfigBuilder()
config_cell.add_column(topic_config)
config_cell.add_column(dd.CustomColumnConfig(name="fact", generator_function=cell_by_cell_generator))

start = time.perf_counter()
result_cell = data_designer.preview(config_builder=config_cell, num_records=NUM_RECORDS)
elapsed_cell = time.perf_counter() - start

# full_column
config_full = dd.DataDesignerConfigBuilder()
config_full.add_column(topic_config)
config_full.add_column(
    dd.CustomColumnConfig(name="fact", generator_function=full_column_generator, generation_strategy="full_column")
)

start = time.perf_counter()
result_full = data_designer.preview(config_builder=config_full, num_records=NUM_RECORDS)
elapsed_full = time.perf_counter() - start

print(f"cell_by_cell: {elapsed_cell:.2f}s | full_column: {elapsed_full:.2f}s")
