# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Multi-turn conversation using two different models.

This demonstrates a Writer/Editor workflow where:
1. Writer model drafts content based on a topic
2. Editor model critiques the draft
3. Writer model revises based on feedback
"""

from __future__ import annotations

import pandas as pd

import data_designer.config as dd
from data_designer.interface import DataDesigner


def multi_turn_writer_editor(df: pd.DataFrame, ctx: dd.CustomColumnContext) -> pd.DataFrame:
    """Multi-turn: Writer drafts → Editor critiques → Writer revises."""

    writer = ctx.get_model("openai-text")
    editor = ctx.get_model("openai-text")

    final_content = []
    conversations = []

    for idx, row in df.iterrows():
        topic = row["topic"]
        print(f"  Processing row {idx}: {topic}")
        conversation = []

        # Turn 1: Writer drafts
        print("    Turn 1: Writer drafting...")
        draft, _ = writer.generate(
            prompt=f"Write a 2-sentence blog hook about '{topic}'.",
            parser=lambda x: x,
            system_prompt="Be creative and concise.",
            max_correction_steps=0,
            max_conversation_restarts=0,
        )
        conversation.append({"role": "writer", "content": draft})
        print("    Turn 1 complete.")

        # Turn 2: Editor critiques
        print("    Turn 2: Editor critiquing...")
        critique, _ = editor.generate(
            prompt=f"Give one specific improvement for this blog hook:\n\n{draft}",
            parser=lambda x: x,
            system_prompt="Be constructive and brief.",
            max_correction_steps=0,
            max_conversation_restarts=0,
        )
        conversation.append({"role": "editor", "content": critique})
        print("    Turn 2 complete.")

        # Turn 3: Writer revises
        print("    Turn 3: Writer revising...")
        revised, _ = writer.generate(
            prompt=f"Revise this hook based on feedback:\n\nOriginal: {draft}\n\nFeedback: {critique}",
            parser=lambda x: x,
            system_prompt="Apply the feedback. Keep it to 2 sentences.",
            max_correction_steps=0,
            max_conversation_restarts=0,
        )
        conversation.append({"role": "writer_revised", "content": revised})
        print(f"    Turn 3 complete. Row {idx} done!")

        final_content.append(revised)
        conversations.append(conversation)

    df[ctx.column_name] = final_content
    df["conversation_trace"] = [str(conv) for conv in conversations]

    return df


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

    print("=" * 70)
    print("MULTI-TURN WRITER/EDITOR EXAMPLE")
    print("3 LLM calls per row × 2 rows = 6 total LLM calls")
    print("=" * 70)

    preview = data_designer.preview(config_builder=config_builder, num_records=2)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for i, row in preview.dataset.iterrows():
        print(f"\n--- Row {i}: '{row['topic']}' ---")
        print(f"FINAL: {row['refined_intro'][:200]}...")
