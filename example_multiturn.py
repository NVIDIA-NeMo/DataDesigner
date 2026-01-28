# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Multi-turn conversation using LLMs.

This demonstrates a Writer/Editor workflow where:
1. Writer model drafts content based on a topic
2. Editor model critiques the draft
3. Writer model revises based on feedback
"""

from __future__ import annotations

import data_designer.config as dd
from data_designer.interface import DataDesigner

WRITER_MODEL_ALIAS = "nvidia-text"
REVIEWER_MODEL_ALIAS = "nvidia-text"


def multi_turn_writer_editor(row: dict, ctx: dd.CustomColumnContext) -> dict:
    """Multi-turn: Writer drafts → Editor critiques → Writer revises."""
    topic = row["topic"]

    # Turn 1: Writer drafts
    draft = ctx.generate_text(
        model_alias=WRITER_MODEL_ALIAS,
        prompt=f"Write a 2-sentence blog hook about '{topic}'.",
        system_prompt="Be creative and concise.",
    )

    # Turn 2: Editor critiques
    critique = ctx.generate_text(
        model_alias=REVIEWER_MODEL_ALIAS,
        prompt=f"Give one specific improvement for this blog hook:\n\n{draft}",
        system_prompt="Be constructive and brief.",
    )

    # Turn 3: Writer revises
    revised = ctx.generate_text(
        model_alias=WRITER_MODEL_ALIAS,
        prompt=f"Revise based on feedback:\n\nOriginal: {draft}\n\nFeedback: {critique}",
        system_prompt="Apply the feedback. Keep it to 2 sentences.",
    )

    row[ctx.column_name] = revised
    row["conversation_trace"] = f"Draft: {draft[:100]}... | Critique: {critique[:100]}..."

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
        dd.CustomColumnConfig(
            name="refined_intro",
            generate_fn=multi_turn_writer_editor,
            input_columns=["topic"],
            output_columns=["conversation_trace"],
        )
    )

    preview = data_designer.preview(config_builder=config_builder, num_records=3)
    preview.display_sample_record()