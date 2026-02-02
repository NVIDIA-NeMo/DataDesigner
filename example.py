# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example: CustomColumnGenerator with decorator-based API."""

from __future__ import annotations

from pydantic import BaseModel

import data_designer.config as dd
from data_designer.interface import DataDesigner

MODEL_ALIAS = "nvidia-text"


class MessageParams(BaseModel):
    tone: str = "friendly"
    max_words: int = 50


# 1-arg: simple transform
@dd.custom_column_generator(required_columns=["name"])
def simple_text_transform(row: dict) -> dict:
    row["greeting"] = f"Hello, {row['name']}!"
    return row


# 2-arg: with typed params
@dd.custom_column_generator(required_columns=["name", "product_interest"])
def greeting_with_params(row: dict, params: MessageParams) -> dict:
    row["greeting"] = f"Hello, {row['name']}! Interested in {row['product_interest']}. (tone: {params.tone})"
    return row


# 3-arg: with params and LLM access
@dd.custom_column_generator(
    required_columns=["name", "product_interest"],
    side_effect_columns=["prompt"],
    model_aliases=[MODEL_ALIAS],
)
def generate_personalized_message(row: dict, params: MessageParams, ctx: dd.CustomColumnContext) -> dict:
    prompt = f"Write a {params.tone} message for {row['name']} interested in {row['product_interest']}. Max {params.max_words} words."
    row[ctx.column_name] = ctx.generate_text(model_alias=MODEL_ALIAS, prompt=prompt)
    row["prompt"] = prompt
    return row


# Direct model access for advanced use cases
@dd.custom_column_generator(required_columns=["product_interest"], model_aliases=[MODEL_ALIAS])
def generate_with_direct_model_access(row: dict, params: None, ctx: dd.CustomColumnContext) -> dict:
    model = ctx.get_model(MODEL_ALIAS)
    response, _ = model.generate(
        prompt=f"Create a tagline for {row['product_interest']} products.",
        parser=lambda x: x,
        system_prompt="You are a creative copywriter.",
        max_correction_steps=0,
        max_conversation_restarts=0,
    )
    row[ctx.column_name] = response
    return row


def develop_with_real_context() -> None:
    """Test generators with real LLM calls during development."""
    data_designer = DataDesigner()
    ctx = dd.CustomColumnContext.from_data_designer(data_designer)
    params = MessageParams(tone="professional", max_words=30)

    result = generate_personalized_message(
        {"name": "Alice", "product_interest": "Electronics"}, params, ctx
    )
    print(result)


def run_full_pipeline() -> None:
    """Run the full DataDesigner pipeline."""
    data_designer = DataDesigner()
    config_builder = dd.DataDesignerConfigBuilder()

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="name",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["Alice", "Bob", "Charlie"]),
        )
    )
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="product_interest",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["Electronics", "Books", "Sports"]),
        )
    )
    config_builder.add_column(
        dd.CustomColumnConfig(name="greeting", generator_function=simple_text_transform)
    )
    config_builder.add_column(
        dd.CustomColumnConfig(
            name="personalized_message",
            generator_function=generate_personalized_message,
            generator_params=MessageParams(tone="professional", max_words=30),
        )
    )

    preview = data_designer.preview(config_builder=config_builder, num_records=5)
    preview.display_sample_record()


if __name__ == "__main__":
    # develop_with_real_context()
    # run_full_pipeline()
    pass
