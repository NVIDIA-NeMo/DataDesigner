# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example demonstrating the CustomColumnGenerator with LLM access.

This example shows how to create a custom column generator that:
1. Uses a simple transformation (no LLM)
2. Uses an LLM model via the CustomColumnContext

To run this example, you need to set up your API key:
    export NVIDIA_API_KEY="your-api-key-here"

Then run:
    uv run python example.py
"""

from __future__ import annotations

import pandas as pd

import data_designer.config as dd
from data_designer.interface import DataDesigner

# =============================================================================
# Example 1: Simple custom generator (no LLM)
# =============================================================================


def simple_text_transform(df: pd.DataFrame) -> pd.DataFrame:
    """A simple generator that transforms text without LLM.

    This function only takes a DataFrame - no context needed for simple transforms.
    """
    df["greeting"] = df["name"].apply(lambda name: f"Hello, {name}! Welcome to our store.")
    return df


# =============================================================================
# Example 2: Custom generator with LLM access via CustomColumnContext
# =============================================================================


def generate_personalized_message(df: pd.DataFrame, ctx: dd.CustomColumnContext) -> pd.DataFrame:
    """A generator that uses an LLM to create personalized messages.

    This demonstrates the clean API provided by CustomColumnContext:
    - ctx.kwargs: Access custom parameters from the config
    - ctx.generate_text_batch(): Parallel text generation with LLMs (fast!)
    - ctx.column_name: The name of the column being generated
    """
    # Access custom kwargs from the config
    tone = ctx.kwargs.get("tone", "friendly")
    max_words = ctx.kwargs.get("max_words", 50)

    # Build all prompts first
    prompts = []
    for _, row in df.iterrows():
        name = row["name"]
        product = row["product_interest"]
        prompt = (
            f"Write a {tone} personalized message for a customer named {name} "
            f"who is interested in {product}. Keep it under {max_words} words."
        )
        prompts.append(prompt)

    # Generate all texts in parallel - much faster!
    results = ctx.generate_text_batch(
        model_alias="nvidia-text",
        prompts=prompts,
        system_prompt="You are a helpful customer service assistant.",
        max_workers=4,  # Adjust based on your API rate limits
    )

    df[ctx.column_name] = results
    return df


# =============================================================================
# Example 3: Advanced - Direct model access for more control
# =============================================================================


def generate_with_direct_model_access(df: pd.DataFrame, ctx: dd.CustomColumnContext) -> pd.DataFrame:
    """Example showing direct model access for advanced use cases.

    Use ctx.get_model() when you need more control over the generation parameters.
    """
    # Get direct model access
    model = ctx.get_model("nvidia-text")

    results = []
    for _, row in df.iterrows():
        prompt = f"Create a catchy tagline for {row['product_interest']} products."

        # Direct model.generate() call with full control
        response, reasoning_trace = model.generate(
            prompt=prompt,
            parser=lambda x: x,  # Return raw text
            system_prompt="You are a creative marketing copywriter.",
            max_correction_steps=0,
            max_conversation_restarts=0,
        )
        results.append(response)

    df[ctx.column_name] = results
    return df


# =============================================================================
# Main example
# =============================================================================


def main() -> None:
    data_designer = DataDesigner()
    config_builder = dd.DataDesignerConfigBuilder()

    # Add a name column using the Category sampler
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="name",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["Alice", "Bob", "Charlie", "Diana", "Eve"],
            ),
        )
    )

    # Add a product interest column
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="product_interest",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["Electronics", "Books", "Home & Garden", "Sports", "Fashion"],
            ),
        )
    )

    # Example 1: Add a simple custom column (no LLM)
    config_builder.add_column(
        dd.CustomColumnConfig(
            name="greeting",
            generate_fn=simple_text_transform,
            input_columns=["name"],
        )
    )

    # Example 2: Add a custom column that uses an LLM via CustomColumnContext
    config_builder.add_column(
        dd.CustomColumnConfig(
            name="personalized_message",
            generate_fn=generate_personalized_message,
            input_columns=["name", "product_interest"],
            kwargs={
                "tone": "friendly and professional",
                "max_words": 30,
            },
        )
    )

    # Preview the dataset (generates a small sample)
    print("Generating preview...")
    preview = data_designer.preview(config_builder=config_builder, num_records=10)

    # Display the results
    print("\n" + "=" * 80)
    print("PREVIEW RESULTS")
    print("=" * 80)

    # The preview result has a `dataset` attribute with the DataFrame
    print(preview.dataset.to_string())

    print("\n" + "=" * 80)
    print("Sample record (formatted):")
    print("=" * 80)
    preview.display_sample_record()


if __name__ == "__main__":
    main()
