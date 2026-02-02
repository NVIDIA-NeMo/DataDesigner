# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: cell_by_cell vs full_column strategies."""

from __future__ import annotations

import time

import pandas as pd

import data_designer.config as dd
from data_designer.interface import DataDesigner

MODEL_ALIAS = "nvidia-text"
NUM_RECORDS = 6


@dd.custom_column_generator(required_columns=["topic"], model_aliases=[MODEL_ALIAS])
def full_column_generator(df: pd.DataFrame, params: None, ctx: dd.CustomColumnContext) -> pd.DataFrame:
    """Batch processing with generate_text_batch."""
    prompts = [f"Write a one-sentence fact about: {topic}" for topic in df["topic"]]
    df["fact"] = ctx.generate_text_batch(model_alias=MODEL_ALIAS, prompts=prompts, max_workers=8)
    return df


@dd.custom_column_generator(required_columns=["topic"], model_aliases=[MODEL_ALIAS])
def cell_by_cell_generator(row: dict, params: None, ctx: dd.CustomColumnContext) -> dict:
    """Row-by-row processing, framework parallelizes."""
    row[ctx.column_name] = ctx.generate_text(
        model_alias=MODEL_ALIAS, prompt=f"Write a one-sentence fact about: {row['topic']}"
    )
    return row


if __name__ == "__main__":
    data_designer = DataDesigner()

    topic_config = dd.SamplerColumnConfig(
        name="topic",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=["space", "history", "biology", "physics"]),
    )

    # full_column
    config_full = dd.DataDesignerConfigBuilder()
    config_full.add_column(topic_config)
    config_full.add_column(
        dd.CustomColumnConfig(name="fact", generator_function=full_column_generator, generation_strategy="full_column")
    )
    start = time.perf_counter()
    result_full = data_designer.preview(config_builder=config_full, num_records=NUM_RECORDS)
    elapsed_full = time.perf_counter() - start

    # cell_by_cell
    config_cell = dd.DataDesignerConfigBuilder()
    config_cell.add_column(topic_config)
    config_cell.add_column(
        dd.CustomColumnConfig(name="fact", generator_function=cell_by_cell_generator)
    )
    start = time.perf_counter()
    result_cell = data_designer.preview(config_builder=config_cell, num_records=NUM_RECORDS)
    elapsed_cell = time.perf_counter() - start

    print(f"full_column: {elapsed_full:.2f}s | cell_by_cell: {elapsed_cell:.2f}s")
