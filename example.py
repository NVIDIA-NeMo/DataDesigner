# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example demonstrating the CustomColumnGenerator with LLM access.
"""

from __future__ import annotations

from pydantic import BaseModel

import data_designer.config as dd
from data_designer.interface import DataDesigner

MODEL_ALIAS = "nvidia-text"


class MessageConfig(BaseModel):
    """Typed configuration for the personalized message generator."""

    tone: str = "friendly"
    max_words: int = 50


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
    - ctx.generator_config: Access typed configuration from the config
    - ctx.generate_text(): Generate text with an LLM
    - ctx.column_name: The name of the column being generated
    """
    config: MessageConfig = ctx.generator_config
    name = row["name"]
    product = row["product_interest"]
    prompt = (
        f"Write a {config.tone} personalized message for a customer named {name} "
        f"who is interested in {product}. Keep it under {config.max_words} words."
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


def develop_with_mock_context() -> None:
    """Develop and iterate on generators using MockCustomColumnContext.

    This lets you test your generator logic without LLM calls or the full framework.
    """
    print("=== Development with MockCustomColumnContext ===\n")

    # Create a mock context with configurable responses
    ctx = dd.MockCustomColumnContext(
        column_name="personalized_message",
        generator_config=MessageConfig(tone="friendly and professional", max_words=30),
        mock_responses=[
            "Hi Alice! Check out our amazing Electronics collection!",
            "Hello Bob! We have great Books just for you!",
        ],
    )

    # Test your generator with sample data
    test_rows = [
        {"name": "Alice", "product_interest": "Electronics"},
        {"name": "Bob", "product_interest": "Books"},
    ]

    for row in test_rows:
        result = generate_personalized_message(row.copy(), ctx)
        print(f"Input:  {row}")
        print(f"Output: {result}")
        print()

    # Inspect what prompts were generated
    print("Generated prompts:")
    for i, call in enumerate(ctx.call_history):
        print(f"  {i + 1}. {call['prompt'][:80]}...")

    print()


def run_full_pipeline() -> None:
    """Run the full DataDesigner pipeline with actual LLM calls."""
    print("=== Full DataDesigner Pipeline ===\n")

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
            generator_function=simple_text_transform,
            input_columns=["name"],
        )
    )

    config_builder.add_column(
        dd.CustomColumnConfig(
            name="personalized_message",
            generator_function=generate_personalized_message,
            input_columns=["name", "product_interest"],
            output_columns=["prompt"],
            model_aliases=[MODEL_ALIAS],
            generator_config=MessageConfig(
                tone="friendly and professional",
                max_words=30,
            ),
        )
    )

    print("Generating preview...")
    preview = data_designer.preview(config_builder=config_builder, num_records=10)
    preview.display_sample_record()


if __name__ == "__main__":
    # First, develop and test with mock context (no LLM calls)
    develop_with_mock_context()

    # Then run the full pipeline (requires API keys and LLM access)
    # Uncomment the line below when ready to test with real LLMs:
    # run_full_pipeline()
