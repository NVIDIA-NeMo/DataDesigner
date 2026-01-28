# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example demonstrating the CustomColumnGenerator with LLM access.
"""

from __future__ import annotations

import data_designer.config as dd
from data_designer.interface import DataDesigner

MODEL_ALIAS = "nvidia-text"


# Example 1: Simple custom generator (no LLM)
def simple_text_transform(row: dict) -> dict:
    """A simple generator that transforms text without LLM.

    This function only takes a row dict - no context needed for simple transforms.
    """
    row["greeting"] = f"Hello, {row['name']}! Welcome to our store."
    return row


# Example 2: Custom generator with LLM access via CustomColumnContext
def generate_personalized_message(row: dict, ctx: dd.CustomColumnContext) -> dict:
    """A generator that uses an LLM to create personalized messages.

    This demonstrates the clean API provided by CustomColumnContext:
    - ctx.kwargs: Access custom parameters from the config
    - ctx.generate_text(): Generate text with an LLM
    - ctx.column_name: The name of the column being generated
    """
    tone = ctx.kwargs.get("tone", "friendly")
    max_words = ctx.kwargs.get("max_words", 50)

    name = row["name"]
    product = row["product_interest"]
    prompt = (
        f"Write a {tone} personalized message for a customer named {name} "
        f"who is interested in {product}. Keep it under {max_words} words."
    )

    result = ctx.generate_text(
        model_alias=MODEL_ALIAS,
        prompt=prompt,
        system_prompt="You are a helpful customer service assistant.",
    )

    row[ctx.column_name] = result
    row["prompt"] = prompt
    return row


# Example 3: Advanced - Direct model access for more control
def generate_with_direct_model_access(row: dict, ctx: dd.CustomColumnContext) -> dict:
    """Example showing direct model access for advanced use cases.

    Use ctx.get_model() when you need more control over the generation parameters.
    """
    model = ctx.get_model(MODEL_ALIAS)

    prompt = f"Create a catchy tagline for {row['product_interest']} products."

    response, _ = model.generate(
        prompt=prompt,
        parser=lambda x: x,  # Return raw text
        system_prompt="You are a creative marketing copywriter.",
        max_correction_steps=0,
        max_conversation_restarts=0,
    )

    row[ctx.column_name] = response
    return row


if __name__ == "__main__":
    data_designer = DataDesigner()
    config_builder = dd.DataDesignerConfigBuilder()

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="name",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["Alice", "Bob", "Charlie", "Diana", "Eve"],
            ),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="product_interest",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["Electronics", "Books", "Home & Garden", "Sports", "Fashion"],
            ),
        )
    )

    config_builder.add_column(
        dd.CustomColumnConfig(
            name="greeting",
            generate_fn=simple_text_transform,
            input_columns=["name"],
        )
    )

    config_builder.add_column(
        dd.CustomColumnConfig(
            name="personalized_message",
            generate_fn=generate_personalized_message,
            input_columns=["name", "product_interest"],
            output_columns=["prompt"],
            kwargs={
                "tone": "friendly and professional",
                "max_words": 30,
            },
        )
    )

    print("Generating preview...")
    preview = data_designer.preview(config_builder=config_builder, num_records=10)
    preview.display_sample_record()
