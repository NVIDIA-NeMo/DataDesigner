# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CustomColumnGenerator with decorator-based API."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from data_designer.config.column_configs import CustomColumnConfig, GenerationStrategy
from data_designer.config.custom_column import CustomColumnContext, custom_column_generator
from data_designer.config.errors import InvalidConfigError
from data_designer.engine.column_generators.generators.custom import CustomColumnGenerator
from data_designer.engine.column_generators.utils.errors import CustomColumnGenerationError
from data_designer.engine.resources.resource_provider import ResourceProvider


class SampleParams(BaseModel):
    """Sample params class for CustomColumnConfig tests."""

    multiplier: int = 1
    prefix: str = ""
    suffix: str = "_processed"


# ============================================================================
# Test fixtures: decorated generator functions
# ============================================================================


@custom_column_generator()
def simple_generator(row: dict) -> dict:
    """Simple 1-arg generator."""
    row["test_column"] = "test_value"
    return row


@custom_column_generator(required_columns=["input"])
def generator_with_required_columns(row: dict) -> dict:
    """Generator that requires input column."""
    row["result"] = row["input"].upper()
    return row


@custom_column_generator(required_columns=["input"])
def generator_with_params(row: dict, params: SampleParams) -> dict:
    """Generator with typed params (2-arg)."""
    row["result"] = f"{params.prefix}{row['input']}{params.suffix}"
    return row


@custom_column_generator(
    required_columns=["input"],
    model_aliases=["test-model"],
)
def generator_with_context(row: dict, params: SampleParams, ctx: CustomColumnContext) -> dict:
    """Generator with params and context (3-arg)."""
    suffix = params.suffix if params else "_processed"
    row["result"] = f"{row['input']}{suffix}"
    return row


@custom_column_generator(
    required_columns=["input"],
    side_effect_columns=["secondary"],
)
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


# ============================================================================
# Decorator tests
# ============================================================================


def test_decorator_stores_metadata() -> None:
    """Test that @custom_column_generator stores metadata on the function."""

    @custom_column_generator(
        required_columns=["a", "b"],
        side_effect_columns=["c"],
        model_aliases=["model-x"],
    )
    def my_func(row: dict) -> dict:
        return row

    assert hasattr(my_func, "_custom_column_metadata")
    metadata = my_func._custom_column_metadata
    assert metadata["required_columns"] == ["a", "b"]
    assert metadata["side_effect_columns"] == ["c"]
    assert metadata["model_aliases"] == ["model-x"]


def test_decorator_default_metadata() -> None:
    """Test that decorator with no args stores empty lists."""

    @custom_column_generator()
    def my_func(row: dict) -> dict:
        return row

    metadata = my_func._custom_column_metadata
    assert metadata["required_columns"] == []
    assert metadata["side_effect_columns"] == []
    assert metadata["model_aliases"] == []


def test_config_reads_from_decorator() -> None:
    """Test that CustomColumnConfig reads metadata from decorator."""

    @custom_column_generator(
        required_columns=["col1", "col2"],
        side_effect_columns=["extra"],
        model_aliases=["model-a"],
    )
    def decorated_generator(row: dict) -> dict:
        return row

    config = CustomColumnConfig(
        name="test",
        generator_function=decorated_generator,
    )

    assert config.required_columns == ["col1", "col2"]
    assert config.side_effect_columns == ["extra"]
    assert config.model_aliases == ["model-a"]


# ============================================================================
# Generator creation tests
# ============================================================================


def test_generator_creation() -> None:
    """Test that a CustomColumnGenerator can be created."""
    generator = _create_test_generator()
    assert generator.config.name == "test_column"
    assert generator.config.column_type == "custom"


def test_generation_strategy_is_cell_by_cell() -> None:
    """Test that CustomColumnGenerator uses CELL_BY_CELL strategy by default."""
    generator = _create_test_generator()
    assert generator.get_generation_strategy() == GenerationStrategy.CELL_BY_CELL


def test_config_validation_non_callable() -> None:
    """Test that non-callable generator_function raises an error."""
    with pytest.raises(InvalidConfigError, match="must be a callable"):
        CustomColumnConfig(
            name="test",
            generator_function="not_a_function",
        )


# ============================================================================
# 1-arg signature tests (simple transform)
# ============================================================================


def test_generate_simple_column() -> None:
    """Test generating a simple column from a row."""
    generator = _create_test_generator(
        name="result",
        generator_function=generator_with_required_columns,
    )

    row = {"input": "hello"}
    result = generator.generate(row)

    assert "result" in result
    assert result["result"] == "HELLO"


def test_generate_missing_required_columns() -> None:
    """Test that missing required columns raise an error."""
    generator = _create_test_generator(
        name="result",
        generator_function=generator_with_required_columns,
    )

    row = {"other_column": 1}

    with pytest.raises(CustomColumnGenerationError, match="Missing required columns"):
        generator.generate(row)


def test_generate_function_raises_error() -> None:
    """Test that errors in the generate function are wrapped."""

    @custom_column_generator()
    def failing_generator(row: dict) -> dict:
        raise ValueError("Something went wrong")

    generator = _create_test_generator(
        name="result",
        generator_function=failing_generator,
    )

    row = {"input": 1}

    with pytest.raises(CustomColumnGenerationError, match="Custom generator function failed"):
        generator.generate(row)


def test_generate_returns_non_dict() -> None:
    """Test that returning non-dict raises an error."""

    @custom_column_generator()
    def wrong_return_type(row: dict) -> list:
        return [1, 2, 3]

    generator = _create_test_generator(
        name="result",
        generator_function=wrong_return_type,
    )

    row = {"input": 1}

    with pytest.raises(CustomColumnGenerationError, match="must return a dict"):
        generator.generate(row)


def test_generate_missing_output_column() -> None:
    """Test that not creating the expected column raises an error."""

    @custom_column_generator()
    def wrong_column_name(row: dict) -> dict:
        row["wrong_column"] = "value"
        return row

    generator = _create_test_generator(
        name="expected_column",
        generator_function=wrong_column_name,
    )

    row = {"input": 1}

    with pytest.raises(CustomColumnGenerationError, match="did not create the expected column"):
        generator.generate(row)


# ============================================================================
# 2-arg signature tests (with params)
# ============================================================================


def test_generate_with_params() -> None:
    """Test that a 2-arg function receives params."""
    params = SampleParams(prefix="hello_", suffix="_world")
    generator = _create_test_generator(
        name="result",
        generator_function=generator_with_params,
        generator_params=params,
    )

    row = {"input": "test"}
    result = generator.generate(row)

    assert result["result"] == "hello_test_world"


def test_generate_with_none_params() -> None:
    """Test that 2-arg function works with None params."""

    @custom_column_generator(required_columns=["input"])
    def generator_handles_none(row: dict, params: SampleParams | None) -> dict:
        suffix = params.suffix if params else "_default"
        row["result"] = f"{row['input']}{suffix}"
        return row

    generator = _create_test_generator(
        name="result",
        generator_function=generator_handles_none,
        generator_params=None,
    )

    row = {"input": "test"}
    result = generator.generate(row)

    assert result["result"] == "test_default"


# ============================================================================
# 3-arg signature tests (with params and context)
# ============================================================================


def test_generate_with_context() -> None:
    """Test that a 3-arg function receives params and context."""
    received_context = None

    @custom_column_generator(required_columns=["input"])
    def capture_context(row: dict, params: SampleParams, ctx: CustomColumnContext) -> dict:
        nonlocal received_context
        received_context = ctx
        suffix = params.suffix if params else "_processed"
        row["result"] = f"{row['input']}{suffix}"
        return row

    params = SampleParams(suffix="_custom")
    generator = _create_test_generator(
        name="result",
        generator_function=capture_context,
        generator_params=params,
    )

    row = {"input": "a"}
    result = generator.generate(row)

    # Verify the context was passed and has correct properties
    assert received_context is not None
    assert isinstance(received_context, CustomColumnContext)
    assert received_context.column_name == "result"

    # Verify the result
    assert result["result"] == "a_custom"


def test_context_provides_model_registry_access() -> None:
    """Test that CustomColumnContext provides access to model_registry."""
    accessed_model_registry = None

    @custom_column_generator()
    def access_registry(row: dict, params: None, ctx: CustomColumnContext) -> dict:
        nonlocal accessed_model_registry
        accessed_model_registry = ctx.model_registry
        row["result"] = "processed"
        return row

    mock_resource_provider = Mock(spec=ResourceProvider)
    mock_model_registry = Mock()
    mock_resource_provider.model_registry = mock_model_registry

    generator = _create_test_generator(
        name="result",
        generator_function=access_registry,
        resource_provider=mock_resource_provider,
    )

    row = {"input": 1}
    generator.generate(row)

    # Verify the model_registry was accessible via context
    assert accessed_model_registry is mock_model_registry


def test_context_generate_text() -> None:
    """Test that CustomColumnContext.generate_text calls the model correctly."""

    @custom_column_generator(
        required_columns=["input"],
        model_aliases=["test-model"],
    )
    def llm_generator(row: dict, params: None, ctx: CustomColumnContext) -> dict:
        text = ctx.generate_text(
            model_alias="test-model",
            prompt=f"Process: {row['input']}",
            system_prompt="You are helpful.",
        )
        row["result"] = text
        return row

    # Set up mocks
    mock_resource_provider = Mock(spec=ResourceProvider)
    mock_model_registry = Mock()
    mock_model = Mock()
    mock_model.generate.return_value = ("Generated text", None)
    mock_model_registry.get_model.return_value = mock_model
    mock_resource_provider.model_registry = mock_model_registry

    generator = _create_test_generator(
        name="result",
        generator_function=llm_generator,
        resource_provider=mock_resource_provider,
    )

    row = {"input": "a"}
    result = generator.generate(row)

    # Verify the model was called correctly
    mock_model_registry.get_model.assert_called_with(model_alias="test-model")

    # Verify generate was called with correct parameters
    mock_model.generate.assert_called_once()
    call_kwargs = mock_model.generate.call_args[1]
    assert "Process: a" in call_kwargs["prompt"]
    assert call_kwargs["system_prompt"] == "You are helpful."

    # Verify results
    assert result["result"] == "Generated text"


# ============================================================================
# Side effect columns tests
# ============================================================================


def test_generate_multiple_side_effect_columns() -> None:
    """Test generating multiple side effect columns."""
    generator = _create_test_generator(
        name="primary",
        generator_function=generator_with_side_effects,
    )

    row = {"input": 5}
    result = generator.generate(row)

    assert "primary" in result
    assert "secondary" in result
    assert result["primary"] == 10
    assert result["secondary"] == 15


def test_missing_declared_side_effect_columns_raises_error() -> None:
    """Test that missing declared side_effect_columns raises an error."""

    @custom_column_generator(
        required_columns=["input"],
        side_effect_columns=["secondary"],
    )
    def missing_side_effect(row: dict) -> dict:
        row["primary"] = row["input"] * 2
        # Missing "secondary" column that was declared
        return row

    generator = _create_test_generator(
        name="primary",
        generator_function=missing_side_effect,
    )

    row = {"input": 1}

    with pytest.raises(CustomColumnGenerationError, match="did not create declared side_effect_columns"):
        generator.generate(row)


def test_undeclared_columns_removed_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Test that undeclared columns are removed with a warning."""
    import logging

    @custom_column_generator(required_columns=["input"])
    def creates_undeclared(row: dict) -> dict:
        row["primary"] = row["input"] * 2
        row["undeclared_column"] = "should be removed"
        return row

    generator = _create_test_generator(
        name="primary",
        generator_function=creates_undeclared,
    )

    row = {"input": 3}

    with caplog.at_level(logging.WARNING):
        result = generator.generate(row)

    # Primary column should exist
    assert "primary" in result
    assert result["primary"] == 6

    # Undeclared column should be removed
    assert "undeclared_column" not in result

    # Warning should be logged
    assert "undeclared columns" in caplog.text
    assert "undeclared_column" in caplog.text


def test_declared_side_effect_columns_kept() -> None:
    """Test that declared side_effect_columns are kept in the result."""

    @custom_column_generator(
        required_columns=["input"],
        side_effect_columns=["secondary", "tertiary"],
    )
    def multi_output(row: dict) -> dict:
        row["primary"] = row["input"] * 2
        row["secondary"] = row["input"] * 3
        row["tertiary"] = row["input"] * 4
        return row

    generator = _create_test_generator(
        name="primary",
        generator_function=multi_output,
    )

    row = {"input": 2}
    result = generator.generate(row)

    # All declared columns should exist
    assert "primary" in result
    assert "secondary" in result
    assert "tertiary" in result
    assert result["primary"] == 4
    assert result["secondary"] == 6
    assert result["tertiary"] == 8


def test_removing_pre_existing_columns_raises_error() -> None:
    """Test that removing pre-existing columns raises an error."""

    @custom_column_generator(required_columns=["input"])
    def removes_column(row: dict) -> dict:
        row["result"] = row["input"] * 2
        del row["input"]  # Remove a pre-existing column
        return row

    generator = _create_test_generator(
        name="result",
        generator_function=removes_column,
    )

    row = {"input": 5, "other": "keep"}

    with pytest.raises(CustomColumnGenerationError, match="removed pre-existing columns"):
        generator.generate(row)


# ============================================================================
# Logging tests
# ============================================================================


def test_log_pre_generation(caplog: pytest.LogCaptureFixture) -> None:
    """Test that log_pre_generation logs correctly."""
    import logging

    @custom_column_generator(
        required_columns=["input"],
        side_effect_columns=["extra"],
        model_aliases=["test-model"],
    )
    def logged_generator(row: dict) -> dict:
        row["result"] = "value"
        return row

    generator = _create_test_generator(
        name="result",
        generator_function=logged_generator,
        generator_params=SampleParams(prefix="test"),
    )

    with caplog.at_level(logging.INFO):
        generator.log_pre_generation()

    assert "Custom column config for column 'result'" in caplog.text
    assert "logged_generator" in caplog.text
    assert "required_columns" in caplog.text
    assert "model_aliases" in caplog.text


def test_config_serialization() -> None:
    """Test that the generator_function serializes to its name."""
    config = CustomColumnConfig(
        name="test",
        generator_function=generator_with_required_columns,
    )

    serialized = config.model_dump()
    assert serialized["generator_function"] == "generator_with_required_columns"


# ============================================================================
# Full column strategy tests
# ============================================================================


def test_full_column_strategy() -> None:
    """Test that full_column strategy processes DataFrame instead of dict."""
    import pandas as pd

    @custom_column_generator(required_columns=["input"])
    def batch_processor(df: pd.DataFrame) -> pd.DataFrame:
        df["result"] = df["input"] * 2
        return df

    generator = _create_test_generator(
        name="result",
        generator_function=batch_processor,
        generation_strategy=GenerationStrategy.FULL_COLUMN,
    )

    # Verify strategy is FULL_COLUMN
    assert generator.get_generation_strategy() == GenerationStrategy.FULL_COLUMN

    # Test with DataFrame
    df = pd.DataFrame({"input": [1, 2, 3]})
    result = generator.generate(df)

    assert isinstance(result, pd.DataFrame)
    assert "result" in result.columns
    assert list(result["result"]) == [2, 4, 6]


def test_full_column_with_params_and_context() -> None:
    """Test full_column strategy with params and context access."""
    import pandas as pd

    @custom_column_generator(required_columns=["input"])
    def batch_with_context(df: pd.DataFrame, params: SampleParams, ctx: CustomColumnContext) -> pd.DataFrame:
        multiplier = params.multiplier if params else 1
        df["result"] = df["input"] * multiplier
        return df

    mock_resource_provider = Mock(spec=ResourceProvider)

    generator = _create_test_generator(
        name="result",
        generator_function=batch_with_context,
        generator_params=SampleParams(multiplier=10),
        resource_provider=mock_resource_provider,
        generation_strategy=GenerationStrategy.FULL_COLUMN,
    )

    df = pd.DataFrame({"input": [1, 2, 3]})
    result = generator.generate(df)

    assert list(result["result"]) == [10, 20, 30]


def test_full_column_validates_output() -> None:
    """Test that full_column validates output columns like cell_by_cell."""
    import pandas as pd

    @custom_column_generator(required_columns=["input"])
    def batch_wrong_column(df: pd.DataFrame) -> pd.DataFrame:
        df["wrong_name"] = df["input"]
        return df

    generator = _create_test_generator(
        name="expected_name",
        generator_function=batch_wrong_column,
        generation_strategy=GenerationStrategy.FULL_COLUMN,
    )

    df = pd.DataFrame({"input": [1, 2, 3]})

    with pytest.raises(CustomColumnGenerationError, match="did not create the expected column"):
        generator.generate(df)


def test_full_column_removes_undeclared_columns(caplog: pytest.LogCaptureFixture) -> None:
    """Test that full_column removes undeclared columns with warning."""
    import logging

    import pandas as pd

    @custom_column_generator(required_columns=["input"])
    def batch_extra_columns(df: pd.DataFrame) -> pd.DataFrame:
        df["result"] = df["input"] * 2
        df["undeclared"] = "should be removed"
        return df

    generator = _create_test_generator(
        name="result",
        generator_function=batch_extra_columns,
        generation_strategy=GenerationStrategy.FULL_COLUMN,
    )

    df = pd.DataFrame({"input": [1, 2, 3]})

    with caplog.at_level(logging.WARNING):
        result = generator.generate(df)

    assert "result" in result.columns
    assert "undeclared" not in result.columns
    assert "undeclared columns" in caplog.text


def test_generate_text_batch() -> None:
    """Test that generate_text_batch parallelizes LLM calls."""
    import pandas as pd

    @custom_column_generator(
        required_columns=["input"],
        model_aliases=["test-model"],
    )
    def batch_with_parallel_llm(df: pd.DataFrame, params: None, ctx: CustomColumnContext) -> pd.DataFrame:
        prompts = [f"Process: {val}" for val in df["input"]]
        results = ctx.generate_text_batch(
            model_alias="test-model",
            prompts=prompts,
            system_prompt="Be helpful.",
            max_workers=4,
        )
        df["result"] = results
        return df

    # Set up mocks
    mock_resource_provider = Mock(spec=ResourceProvider)
    mock_model_registry = Mock()
    mock_model = Mock()
    mock_model.generate.side_effect = lambda prompt, **kwargs: (f"Response for: {prompt}", None)
    mock_model_registry.get_model.return_value = mock_model
    mock_resource_provider.model_registry = mock_model_registry

    generator = _create_test_generator(
        name="result",
        generator_function=batch_with_parallel_llm,
        resource_provider=mock_resource_provider,
        generation_strategy=GenerationStrategy.FULL_COLUMN,
    )

    df = pd.DataFrame({"input": ["a", "b", "c"]})
    result = generator.generate(df)

    # Verify results
    assert list(result["result"]) == [
        "Response for: Process: a",
        "Response for: Process: b",
        "Response for: Process: c",
    ]

    # Verify model.generate was called 3 times (once per prompt)
    assert mock_model.generate.call_count == 3


# ============================================================================
# Multi-step workflow tests
# ============================================================================


def test_multi_step_workflow_intermediate_failure() -> None:
    """Test that an intermediate LLM failure in a multi-step workflow is handled correctly."""
    call_count = 0

    @custom_column_generator(
        required_columns=["topic"],
        model_aliases=["writer", "editor"],
    )
    def multi_turn_generator(row: dict, params: None, ctx: CustomColumnContext) -> dict:
        nonlocal call_count

        # Step 1: First LLM call succeeds
        draft = ctx.generate_text(model_alias="writer", prompt="Write draft")
        call_count += 1

        # Step 2: Second LLM call fails
        critique = ctx.generate_text(model_alias="editor", prompt=f"Critique: {draft}")
        call_count += 1

        # Step 3: Would use critique but never reached
        row["result"] = critique
        return row

    # Set up mock where second call fails
    mock_resource_provider = Mock(spec=ResourceProvider)
    mock_model_registry = Mock()

    mock_model = Mock()
    call_counter = {"count": 0}

    def mock_generate(**kwargs: dict) -> tuple[str, None]:
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            return ("First draft text", None)
        else:
            raise RuntimeError("LLM API error: rate limited")

    mock_model.generate.side_effect = mock_generate
    mock_model_registry.get_model.return_value = mock_model
    mock_resource_provider.model_registry = mock_model_registry

    generator = _create_test_generator(
        name="result",
        generator_function=multi_turn_generator,
        resource_provider=mock_resource_provider,
    )

    row = {"topic": "testing"}

    # The error from the second LLM call should propagate
    with pytest.raises(CustomColumnGenerationError, match="Custom generator function failed"):
        generator.generate(row)

    # Verify first call succeeded before failure
    assert call_counter["count"] == 2
