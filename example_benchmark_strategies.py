# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark: cell_by_cell vs full_column strategies for CustomColumnGenerator.

This example compares:
- full_column: User controls parallelization via generate_text_batch()
- cell_by_cell: Framework parallelizes across rows automatically

Run with: uv run python example_benchmark_strategies.py
"""

from __future__ import annotations

import time

import pandas as pd

import data_designer.config as dd
from data_designer.interface import DataDesigner

MODEL_ALIAS = "nvidia-text"
NUM_RECORDS = 6


# Strategy 1: full_column - user parallelizes via generate_text_batch
def full_column_generator(df: pd.DataFrame, ctx: dd.CustomColumnContext) -> pd.DataFrame:
    """Entire batch processed at once, user parallelizes with generate_text_batch."""
    prompts = [f"Write a one-sentence fact about the topic: {topic}" for topic in df["topic"]]

    results = ctx.generate_text_batch(
        model_alias=MODEL_ALIAS,
        prompts=prompts,
        system_prompt="Be concise and informative.",
        max_workers=8,
    )

    df["fact"] = results
    return df


# Strategy 2: cell_by_cell (default) - framework parallelizes
def cell_by_cell_generator(row: dict, ctx: dd.CustomColumnContext) -> dict:
    """Each row processed independently, framework handles parallelization."""
    response = ctx.generate_text(
        model_alias=MODEL_ALIAS,
        prompt=f"Write a one-sentence fact about the topic: {row['topic']}",
        system_prompt="Be concise and informative.",
    )
    row[ctx.column_name] = response
    return row


if __name__ == "__main__":
    data_designer = DataDesigner()

    # Common sampler column
    topic_config = dd.SamplerColumnConfig(
        name="topic",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["space exploration", "ancient history", "marine biology", "quantum physics"]
        ),
    )

    # Benchmark full_column first
    print(f"\n{'=' * 60}")
    print("Strategy: full_column (user parallelization via generate_text_batch)")
    print(f"Records: {NUM_RECORDS}")
    print(f"{'=' * 60}")

    config_full = dd.DataDesignerConfigBuilder()
    config_full.add_column(topic_config)
    config_full.add_column(
        dd.CustomColumnConfig(
            name="fact",
            generate_fn=full_column_generator,
            input_columns=["topic"],
            generation_strategy="full_column",
        )
    )

    start = time.perf_counter()
    result_full = data_designer.preview(config_builder=config_full, num_records=NUM_RECORDS)
    elapsed_full = time.perf_counter() - start

    print(f"Time: {elapsed_full:.2f}s")
    print(f"Sample result: {result_full.dataset['fact'].iloc[0][:80]}...")

    # Benchmark cell_by_cell
    print(f"\n{'=' * 60}")
    print("Strategy: cell_by_cell (framework parallelization)")
    print(f"Records: {NUM_RECORDS}")
    print(f"{'=' * 60}")

    config_cell = dd.DataDesignerConfigBuilder()
    config_cell.add_column(topic_config)
    config_cell.add_column(
        dd.CustomColumnConfig(
            name="fact",
            generate_fn=cell_by_cell_generator,
            input_columns=["topic"],
            generation_strategy="cell_by_cell",
        )
    )

    start = time.perf_counter()
    result_cell = data_designer.preview(config_builder=config_cell, num_records=NUM_RECORDS)
    elapsed_cell = time.perf_counter() - start

    print(f"Time: {elapsed_cell:.2f}s")
    print(f"Sample result: {result_cell.dataset['fact'].iloc[0][:80]}...")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"full_column:  {elapsed_full:.2f}s")
    print(f"cell_by_cell: {elapsed_cell:.2f}s")

    if elapsed_cell < elapsed_full:
        speedup = elapsed_full / elapsed_cell
        print(f"\ncell_by_cell is {speedup:.1f}x faster")
    else:
        speedup = elapsed_cell / elapsed_full
        print(f"\nfull_column is {speedup:.1f}x faster")