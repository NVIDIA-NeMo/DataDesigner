# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example: Multi-turn conversation (Writer/Editor workflow)."""

from __future__ import annotations

import data_designer.config as dd
from data_designer.interface import DataDesigner

WRITER = "nvidia-text"
EDITOR = "nvidia-text"


@dd.custom_column_generator(
    required_columns=["topic"],
    side_effect_columns=["conversation_trace"],
    model_aliases=[WRITER, EDITOR],
)
def multi_turn_writer_editor(row: dict, params: None, ctx: dd.CustomColumnContext) -> dict:
    """Writer drafts -> Editor critiques -> Writer revises."""
    topic = row["topic"]

    draft = ctx.generate_text(model_alias=WRITER, prompt=f"Write a 2-sentence hook about '{topic}'.")
    critique = ctx.generate_text(model_alias=EDITOR, prompt=f"Give one improvement for: {draft}")
    revised = ctx.generate_text(model_alias=WRITER, prompt=f"Revise based on feedback:\n\nOriginal: {draft}\n\nFeedback: {critique}")

    row[ctx.column_name] = revised
    row["conversation_trace"] = f"Draft: {draft[:50]}... | Critique: {critique[:50]}..."
    return row


if __name__ == "__main__":
    data_designer = DataDesigner()
    config_builder = dd.DataDesignerConfigBuilder()

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="topic",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["sustainable living", "remote work", "learning languages"]),
        )
    )
    config_builder.add_column(
        dd.CustomColumnConfig(name="refined_intro", generator_function=multi_turn_writer_editor)
    )

    preview = data_designer.preview(config_builder=config_builder, num_records=3)
    preview.display_sample_record()
