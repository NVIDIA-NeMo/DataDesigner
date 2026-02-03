# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CustomColumnGenerator with decorator-based API."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
from pydantic import BaseModel

from data_designer.config.column_configs import CustomColumnConfig, GenerationStrategy
from data_designer.config.custom_column import CustomColumnContext, custom_column_generator
from data_designer.config.errors import InvalidConfigError
from data_designer.engine.column_generators.generators.custom import CustomColumnGenerator
from data_designer.engine.column_generators.utils.errors import CustomColumnGenerationError
from data_designer.engine.resources.resource_provider import ResourceProvider


class SampleParams(BaseModel):
    """Sample params class for tests."""

    multiplier: int = 1
    prefix: str = ""
    suffix: str = "_processed"


# Test fixtures


@custom_column_generator(required_columns=["input"])
def generator_with_required_columns(row: dict) -> dict:
    """Generator that requires input column."""
    row["result"] = row["input"].upper()
    return row


@custom_column_generator(required_columns=["input"], side_effect_columns=["secondary"])
def generator_with_side_effects(row: dict) -> dict:
    """Generator that creates additional columns."""
    row["primary"] = row["input"] * 2
    row["secondary"] = row["input"] * 3
    return row


def _create_test_generator(
    name: str = "test_column",
    generator_function: Any = None,
    generator_params: BaseModel | None = None,
    resource_provider: ResourceProvider | None = None,
    generation_strategy: GenerationStrategy = GenerationStrategy.CELL_BY_CELL,
) -> CustomColumnGenerator:
    """Helper function to create test generator."""
    if generator_function is None:

        @custom_column_generator()
        def simple_generator(row: dict) -> dict:
            row[name] = "test_value"
            return row

        generator_function = simple_generator

    config = CustomColumnConfig(
        name=name,
        generator_function=generator_function,
        generator_params=generator_params,
        generation_strategy=generation_strategy,
    )
    if resource_provider is None:
        resource_provider = Mock(spec=ResourceProvider)
    return CustomColumnGenerator(config=config, resource_provider=resource_provider)


# Config and creation tests


def test_config_and_decorator_integration() -> None:
    """Test config reads decorator metadata, serializes correctly, and creates generator."""

    @custom_column_generator(
        required_columns=["col1", "col2"],
        side_effect_columns=["extra"],
        model_aliases=["model-a"],
    )
    def decorated_generator(row: dict) -> dict:
        return row

    config = CustomColumnConfig(name="test", generator_function=decorated_generator)

    # Decorator metadata is read
    assert config.required_columns == ["col1", "col2"]
    assert config.side_effect_columns == ["extra"]
    assert config.model_aliases == ["model-a"]

    # Serialization works
    assert config.model_dump()["generator_function"] == "decorated_generator"

    # Generator creation works with defaults
    generator = CustomColumnGenerator(config=config, resource_provider=Mock(spec=ResourceProvider))
    assert generator.config.column_type == "custom"
    assert generator.get_generation_strategy() == GenerationStrategy.CELL_BY_CELL


def test_config_validation_non_callable() -> None:
    """Test that non-callable generator_function raises an error."""
    with pytest.raises(InvalidConfigError, match="must be a callable"):
        CustomColumnConfig(name="test", generator_function="not_a_function")


# Cell-by-cell generation tests


def test_cell_by_cell_generation() -> None:
    """Test basic cell-by-cell generation with 1-arg function."""
    generator = _create_test_generator(name="result", generator_function=generator_with_required_columns)
    result = generator.generate({"input": "hello"})
    assert result["result"] == "HELLO"


def test_cell_by_cell_with_params_and_context(stub_resource_provider, stub_model_facade) -> None:
    """Test 3-arg function with params and context for LLM access."""

    @custom_column_generator(required_columns=["input"], model_aliases=["test-model"])
    def llm_generator(row: dict, params: SampleParams, ctx: CustomColumnContext) -> dict:
        text = ctx.generate_text(
            model_alias="test-model",
            prompt=f"{params.prefix}{row['input']}",
            system_prompt="You are helpful.",
        )
        row["result"] = text
        return row

    generator = _create_test_generator(
        name="result",
        generator_function=llm_generator,
        generator_params=SampleParams(prefix="Process: "),
        resource_provider=stub_resource_provider,
    )

    result = generator.generate({"input": "test"})

    # Model was called with correct params
    stub_model_facade.generate.assert_called_once()
    call_kwargs = stub_model_facade.generate.call_args[1]
    assert "Process: test" in call_kwargs["prompt"]
    assert result["result"] == "Generated summary text"


def test_side_effect_columns() -> None:
    """Test that declared side_effect_columns are created and kept."""
    generator = _create_test_generator(name="primary", generator_function=generator_with_side_effects)
    result = generator.generate({"input": 5})

    assert result["primary"] == 10
    assert result["secondary"] == 15


# Error handling tests


@pytest.mark.parametrize(
    "generator_fn,input_row,error_match",
    [
        # Missing required column
        (generator_with_required_columns, {"other": 1}, "Missing required columns"),
        # Function raises error
        (
            custom_column_generator()(lambda row: (_ for _ in ()).throw(ValueError("fail"))),
            {"input": 1},
            "Custom generator function failed",
        ),
    ],
    ids=["missing_required", "function_raises"],
)
def test_generation_errors(generator_fn, input_row, error_match) -> None:
    """Test various error conditions during generation."""
    generator = _create_test_generator(name="result", generator_function=generator_fn)
    with pytest.raises(CustomColumnGenerationError, match=error_match):
        generator.generate(input_row)


def test_output_validation_errors() -> None:
    """Test output validation: wrong return type, missing column, missing side effects."""

    # Wrong return type
    @custom_column_generator()
    def returns_list(row: dict) -> list:
        return [1, 2, 3]

    generator = _create_test_generator(name="result", generator_function=returns_list)
    with pytest.raises(CustomColumnGenerationError, match="must return a dict"):
        generator.generate({"input": 1})

    # Missing expected column
    @custom_column_generator()
    def wrong_column(row: dict) -> dict:
        row["wrong"] = "value"
        return row

    generator = _create_test_generator(name="expected", generator_function=wrong_column)
    with pytest.raises(CustomColumnGenerationError, match="did not create the expected column"):
        generator.generate({"input": 1})

    # Missing declared side effect
    @custom_column_generator(side_effect_columns=["secondary"])
    def missing_side_effect(row: dict) -> dict:
        row["primary"] = 1
        return row

    generator = _create_test_generator(name="primary", generator_function=missing_side_effect)
    with pytest.raises(CustomColumnGenerationError, match="did not create declared side_effect_columns"):
        generator.generate({"input": 1})


def test_undeclared_columns_removed_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Test that undeclared columns are removed with a warning."""
    import logging

    @custom_column_generator()
    def creates_undeclared(row: dict) -> dict:
        row["result"] = "value"
        row["undeclared"] = "should be removed"
        return row

    generator = _create_test_generator(name="result", generator_function=creates_undeclared)

    with caplog.at_level(logging.WARNING):
        result = generator.generate({"input": 1})

    assert "result" in result
    assert "undeclared" not in result
    assert "undeclared columns" in caplog.text


# Full column strategy tests


def test_full_column_strategy() -> None:
    """Test full_column strategy processes DataFrame."""

    @custom_column_generator(required_columns=["input"])
    def batch_processor(df: pd.DataFrame) -> pd.DataFrame:
        df["result"] = df["input"] * 2
        return df

    generator = _create_test_generator(
        name="result", generator_function=batch_processor, generation_strategy=GenerationStrategy.FULL_COLUMN
    )

    assert generator.get_generation_strategy() == GenerationStrategy.FULL_COLUMN

    result = generator.generate(pd.DataFrame({"input": [1, 2, 3]}))
    assert list(result["result"]) == [2, 4, 6]


def test_full_column_with_batch_llm(stub_resource_provider, stub_model_facade) -> None:
    """Test full_column with generate_text_batch for parallel LLM calls."""

    @custom_column_generator(required_columns=["input"], model_aliases=["test-model"])
    def batch_llm(df: pd.DataFrame, params: None, ctx: CustomColumnContext) -> pd.DataFrame:
        prompts = [f"Process: {val}" for val in df["input"]]
        df["result"] = ctx.generate_text_batch(model_alias="test-model", prompts=prompts, max_workers=2)
        return df

    stub_model_facade.generate.side_effect = lambda prompt, **kwargs: (f"Response: {prompt}", None)

    generator = _create_test_generator(
        name="result",
        generator_function=batch_llm,
        resource_provider=stub_resource_provider,
        generation_strategy=GenerationStrategy.FULL_COLUMN,
    )

    result = generator.generate(pd.DataFrame({"input": ["a", "b"]}))

    assert list(result["result"]) == ["Response: Process: a", "Response: Process: b"]
    assert stub_model_facade.generate.call_count == 2
