# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CustomColumnGenerator with CELL_BY_CELL strategy."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.custom_column import CustomColumnContext
from data_designer.config.errors import InvalidConfigError
from data_designer.engine.column_generators.generators.base import GenerationStrategy
from data_designer.engine.column_generators.generators.custom import CustomColumnGenerator
from data_designer.engine.column_generators.utils.errors import CustomColumnGenerationError
from data_designer.engine.resources.resource_provider import ResourceProvider


def _create_test_generator(
    name: str = "test_column",
    generate_fn: callable = None,
    input_columns: list[str] | None = None,
    output_columns: list[str] | None = None,
    kwargs: dict | None = None,
    resource_provider: ResourceProvider | None = None,
) -> CustomColumnGenerator:
    """Helper function to create test generator."""
    if generate_fn is None:

        def generate_fn(row: dict) -> dict:
            row[name] = "test_value"
            return row

    config = CustomColumnConfig(
        name=name,
        generate_fn=generate_fn,
        input_columns=input_columns or [],
        output_columns=output_columns or [],
        kwargs=kwargs or {},
    )
    if resource_provider is None:
        resource_provider = Mock(spec=ResourceProvider)
    return CustomColumnGenerator(config=config, resource_provider=resource_provider)


def test_generator_creation() -> None:
    """Test that a CustomColumnGenerator can be created."""
    generator = _create_test_generator()
    assert generator.config.name == "test_column"
    assert generator.config.column_type == "custom"


def test_generation_strategy_is_cell_by_cell() -> None:
    """Test that CustomColumnGenerator uses CELL_BY_CELL strategy."""
    generator = _create_test_generator()
    assert generator.get_generation_strategy() == GenerationStrategy.CELL_BY_CELL


def test_generate_simple_column() -> None:
    """Test generating a simple column from a row."""

    def my_generator(row: dict) -> dict:
        row["result"] = row["input"].upper()
        return row

    generator = _create_test_generator(
        name="result",
        generate_fn=my_generator,
        input_columns=["input"],
    )

    row = {"input": "hello"}
    result = generator.generate(row)

    assert "result" in result
    assert result["result"] == "HELLO"


def test_generate_with_kwargs() -> None:
    """Test that kwargs are accessible via config."""

    def my_generator(row: dict) -> dict:
        row["result"] = "processed"
        return row

    generator = _create_test_generator(
        name="result",
        generate_fn=my_generator,
        kwargs={"multiplier": 2, "prefix": "test_"},
    )

    assert generator.config.kwargs == {"multiplier": 2, "prefix": "test_"}


def test_generate_missing_required_columns() -> None:
    """Test that missing required columns raise an error."""

    def my_generator(row: dict) -> dict:
        row["result"] = row["missing_column"]
        return row

    generator = _create_test_generator(
        name="result",
        generate_fn=my_generator,
        input_columns=["missing_column"],
    )

    row = {"other_column": 1}

    with pytest.raises(CustomColumnGenerationError, match="Missing required columns"):
        generator.generate(row)


def test_generate_function_raises_error() -> None:
    """Test that errors in the generate function are wrapped."""

    def my_generator(row: dict) -> dict:
        raise ValueError("Something went wrong")

    generator = _create_test_generator(
        name="result",
        generate_fn=my_generator,
    )

    row = {"input": 1}

    with pytest.raises(CustomColumnGenerationError, match="Custom generator function failed"):
        generator.generate(row)


def test_generate_returns_non_dict() -> None:
    """Test that returning non-dict raises an error."""

    def my_generator(row: dict) -> list:
        return [1, 2, 3]

    generator = _create_test_generator(
        name="result",
        generate_fn=my_generator,
    )

    row = {"input": 1}

    with pytest.raises(CustomColumnGenerationError, match="must return a dict"):
        generator.generate(row)


def test_generate_missing_output_column() -> None:
    """Test that not creating the expected column raises an error."""

    def my_generator(row: dict) -> dict:
        row["wrong_column"] = "value"
        return row

    generator = _create_test_generator(
        name="expected_column",
        generate_fn=my_generator,
    )

    row = {"input": 1}

    with pytest.raises(CustomColumnGenerationError, match="did not create the expected column"):
        generator.generate(row)


def test_config_validation_non_callable() -> None:
    """Test that non-callable generate_fn raises an error."""
    with pytest.raises(InvalidConfigError, match="must be a callable"):
        CustomColumnConfig(
            name="test",
            generate_fn="not_a_function",
        )


def test_config_required_columns_property() -> None:
    """Test that required_columns returns input_columns."""

    def my_generator(row: dict) -> dict:
        return row

    config = CustomColumnConfig(
        name="test",
        generate_fn=my_generator,
        input_columns=["col1", "col2"],
    )

    assert config.required_columns == ["col1", "col2"]


def test_config_side_effect_columns_property() -> None:
    """Test that side_effect_columns returns output_columns."""

    def my_generator(row: dict) -> dict:
        return row

    config = CustomColumnConfig(
        name="test",
        generate_fn=my_generator,
        output_columns=["extra_col1", "extra_col2"],
    )

    assert config.side_effect_columns == ["extra_col1", "extra_col2"]


def test_generate_multiple_output_columns() -> None:
    """Test generating multiple output columns."""

    def my_generator(row: dict) -> dict:
        row["primary"] = row["input"] * 2
        row["secondary"] = row["input"] * 3
        return row

    generator = _create_test_generator(
        name="primary",
        generate_fn=my_generator,
        input_columns=["input"],
        output_columns=["secondary"],
    )

    row = {"input": 5}
    result = generator.generate(row)

    assert "primary" in result
    assert "secondary" in result
    assert result["primary"] == 10
    assert result["secondary"] == 15


def test_log_pre_generation(caplog: pytest.LogCaptureFixture) -> None:
    """Test that log_pre_generation logs correctly."""
    import logging

    def my_generator(row: dict) -> dict:
        row["result"] = "value"
        return row

    generator = _create_test_generator(
        name="result",
        generate_fn=my_generator,
        input_columns=["input"],
        output_columns=["extra"],
        kwargs={"key": "value"},
    )

    with caplog.at_level(logging.INFO):
        generator.log_pre_generation()

    assert "Custom column config for column 'result'" in caplog.text
    assert "my_generator" in caplog.text
    assert "input_columns" in caplog.text


def test_config_serialization() -> None:
    """Test that the generate_fn serializes to its name."""

    def my_custom_function(row: dict) -> dict:
        return row

    config = CustomColumnConfig(
        name="test",
        generate_fn=my_custom_function,
    )

    serialized = config.model_dump()
    assert serialized["generate_fn"] == "my_custom_function"


def test_generate_with_context_access() -> None:
    """Test that a two-argument function receives a CustomColumnContext."""
    received_context = None

    def my_generator_with_access(row: dict, ctx: CustomColumnContext) -> dict:
        nonlocal received_context
        received_context = ctx
        # Access kwargs through the context
        suffix = ctx.kwargs.get("suffix", "_processed")
        row["result"] = f"{row['input']}{suffix}"
        return row

    generator = _create_test_generator(
        name="result",
        generate_fn=my_generator_with_access,
        input_columns=["input"],
        kwargs={"suffix": "_custom"},
    )

    row = {"input": "a"}
    result = generator.generate(row)

    # Verify the context was passed and has correct properties
    assert received_context is not None
    assert isinstance(received_context, CustomColumnContext)
    assert received_context.kwargs == {"suffix": "_custom"}
    assert received_context.column_name == "result"

    # Verify the result
    assert result["result"] == "a_custom"


def test_context_provides_model_registry_access() -> None:
    """Test that CustomColumnContext provides access to model_registry."""
    accessed_model_registry = None

    def my_generator_with_resources(row: dict, ctx: CustomColumnContext) -> dict:
        nonlocal accessed_model_registry
        accessed_model_registry = ctx.model_registry
        row["result"] = "processed"
        return row

    mock_resource_provider = Mock(spec=ResourceProvider)
    mock_model_registry = Mock()
    mock_resource_provider.model_registry = mock_model_registry

    generator = _create_test_generator(
        name="result",
        generate_fn=my_generator_with_resources,
        resource_provider=mock_resource_provider,
    )

    row = {"input": 1}
    generator.generate(row)

    # Verify the model_registry was accessible via context
    assert accessed_model_registry is mock_model_registry


def test_context_generate_text() -> None:
    """Test that CustomColumnContext.generate_text calls the model correctly."""

    def my_generator_with_llm(row: dict, ctx: CustomColumnContext) -> dict:
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
        generate_fn=my_generator_with_llm,
        input_columns=["input"],
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


def test_missing_declared_output_columns_raises_error() -> None:
    """Test that missing declared output_columns raises an error."""

    def my_generator(row: dict) -> dict:
        row["primary"] = row["input"] * 2
        # Missing "secondary" column that was declared
        return row

    generator = _create_test_generator(
        name="primary",
        generate_fn=my_generator,
        input_columns=["input"],
        output_columns=["secondary"],
    )

    row = {"input": 1}

    with pytest.raises(CustomColumnGenerationError, match="did not create declared output columns"):
        generator.generate(row)


def test_undeclared_columns_removed_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Test that undeclared columns are removed with a warning."""
    import logging

    def my_generator(row: dict) -> dict:
        row["primary"] = row["input"] * 2
        row["undeclared_column"] = "should be removed"
        return row

    generator = _create_test_generator(
        name="primary",
        generate_fn=my_generator,
        input_columns=["input"],
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


def test_declared_output_columns_kept() -> None:
    """Test that declared output_columns are kept in the result."""

    def my_generator(row: dict) -> dict:
        row["primary"] = row["input"] * 2
        row["secondary"] = row["input"] * 3
        row["tertiary"] = row["input"] * 4
        return row

    generator = _create_test_generator(
        name="primary",
        generate_fn=my_generator,
        input_columns=["input"],
        output_columns=["secondary", "tertiary"],
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


def test_multi_step_workflow_intermediate_failure() -> None:
    """Test that an intermediate LLM failure in a multi-step workflow is handled correctly.

    When a multi-turn workflow fails at an intermediate step (e.g., second LLM call),
    the error should propagate up and be wrapped in CustomColumnGenerationError.
    """
    call_count = 0

    def multi_turn_generator(row: dict, ctx: CustomColumnContext) -> dict:
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
        generate_fn=multi_turn_generator,
        input_columns=["topic"],
        resource_provider=mock_resource_provider,
    )

    row = {"topic": "testing"}

    # The error from the second LLM call should propagate
    with pytest.raises(CustomColumnGenerationError, match="Custom generator function failed"):
        generator.generate(row)

    # Verify first call succeeded before failure
    assert call_counter["count"] == 2
