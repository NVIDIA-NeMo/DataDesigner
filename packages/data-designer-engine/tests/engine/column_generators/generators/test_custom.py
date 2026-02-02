# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CustomColumnGenerator with CELL_BY_CELL strategy."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from data_designer.config.column_configs import CustomColumnConfig, GenerationStrategy
from data_designer.config.custom_column import CustomColumnContext
from data_designer.config.errors import InvalidConfigError
from data_designer.engine.column_generators.generators.custom import CustomColumnGenerator
from data_designer.engine.column_generators.utils.errors import CustomColumnGenerationError
from data_designer.engine.resources.resource_provider import ResourceProvider


class SampleGeneratorConfig(BaseModel):
    """Sample config class for CustomColumnConfig tests."""

    multiplier: int = 1
    prefix: str = ""
    suffix: str = "_processed"


def _create_test_generator(
    name: str = "test_column",
    generator_function: Any = None,
    input_columns: list[str] | None = None,
    output_columns: list[str] | None = None,
    model_aliases: list[str] | None = None,
    generator_config: BaseModel | None = None,
    resource_provider: ResourceProvider | None = None,
) -> CustomColumnGenerator:
    """Helper function to create test generator."""
    if generator_function is None:

        def generator_function(row: dict) -> dict:
            row[name] = "test_value"
            return row

    config = CustomColumnConfig(
        name=name,
        generator_function=generator_function,
        input_columns=input_columns or [],
        output_columns=output_columns or [],
        model_aliases=model_aliases or [],
        generator_config=generator_config,
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
        generator_function=my_generator,
        input_columns=["input"],
    )

    row = {"input": "hello"}
    result = generator.generate(row)

    assert "result" in result
    assert result["result"] == "HELLO"


def test_generate_with_generator_config() -> None:
    """Test that generator_config is accessible via config."""

    def my_generator(row: dict) -> dict:
        row["result"] = "processed"
        return row

    test_config = SampleGeneratorConfig(multiplier=2, prefix="test_")
    generator = _create_test_generator(
        name="result",
        generator_function=my_generator,
        generator_config=test_config,
    )

    assert generator.config.generator_config == test_config
    assert generator.config.generator_config.multiplier == 2
    assert generator.config.generator_config.prefix == "test_"


def test_generate_missing_required_columns() -> None:
    """Test that missing required columns raise an error."""

    def my_generator(row: dict) -> dict:
        row["result"] = row["missing_column"]
        return row

    generator = _create_test_generator(
        name="result",
        generator_function=my_generator,
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
        generator_function=my_generator,
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
        generator_function=my_generator,
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
        generator_function=my_generator,
    )

    row = {"input": 1}

    with pytest.raises(CustomColumnGenerationError, match="did not create the expected column"):
        generator.generate(row)


def test_config_validation_non_callable() -> None:
    """Test that non-callable generator_function raises an error."""
    with pytest.raises(InvalidConfigError, match="must be a callable"):
        CustomColumnConfig(
            name="test",
            generator_function="not_a_function",
        )


def test_config_required_columns_property() -> None:
    """Test that required_columns property returns input_columns."""

    def my_generator(row: dict) -> dict:
        return row

    config = CustomColumnConfig(
        name="test",
        generator_function=my_generator,
        input_columns=["col1", "col2"],
    )

    assert config.required_columns == ["col1", "col2"]
    assert config.input_columns == ["col1", "col2"]


def test_config_side_effect_columns_property() -> None:
    """Test that side_effect_columns property returns output_columns."""

    def my_generator(row: dict) -> dict:
        return row

    config = CustomColumnConfig(
        name="test",
        generator_function=my_generator,
        output_columns=["extra_col1", "extra_col2"],
    )

    assert config.side_effect_columns == ["extra_col1", "extra_col2"]
    assert config.output_columns == ["extra_col1", "extra_col2"]


def test_config_model_aliases_property() -> None:
    """Test that model_aliases field is accessible."""

    def my_generator(row: dict) -> dict:
        return row

    config = CustomColumnConfig(
        name="test",
        generator_function=my_generator,
        model_aliases=["model-a", "model-b"],
    )

    assert config.model_aliases == ["model-a", "model-b"]


def test_generate_multiple_side_effect_columns() -> None:
    """Test generating multiple side effect columns."""

    def my_generator(row: dict) -> dict:
        row["primary"] = row["input"] * 2
        row["secondary"] = row["input"] * 3
        return row

    generator = _create_test_generator(
        name="primary",
        generator_function=my_generator,
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
        generator_function=my_generator,
        input_columns=["input"],
        output_columns=["extra"],
        model_aliases=["test-model"],
        generator_config=SampleGeneratorConfig(prefix="test"),
    )

    with caplog.at_level(logging.INFO):
        generator.log_pre_generation()

    assert "Custom column config for column 'result'" in caplog.text
    assert "my_generator" in caplog.text
    assert "required_columns" in caplog.text
    assert "model_aliases" in caplog.text


def test_config_serialization() -> None:
    """Test that the generator_function serializes to its name."""

    def my_custom_function(row: dict) -> dict:
        return row

    config = CustomColumnConfig(
        name="test",
        generator_function=my_custom_function,
    )

    serialized = config.model_dump()
    assert serialized["generator_function"] == "my_custom_function"


def test_generate_with_context_access() -> None:
    """Test that a two-argument function receives a CustomColumnContext."""
    received_context = None

    def my_generator_with_access(row: dict, ctx: CustomColumnContext) -> dict:
        nonlocal received_context
        received_context = ctx
        # Access generator_config through the context
        suffix = ctx.generator_config.suffix if ctx.generator_config else "_processed"
        row["result"] = f"{row['input']}{suffix}"
        return row

    test_config = SampleGeneratorConfig(suffix="_custom")
    generator = _create_test_generator(
        name="result",
        generator_function=my_generator_with_access,
        input_columns=["input"],
        generator_config=test_config,
    )

    row = {"input": "a"}
    result = generator.generate(row)

    # Verify the context was passed and has correct properties
    assert received_context is not None
    assert isinstance(received_context, CustomColumnContext)
    assert received_context.generator_config == test_config
    assert received_context.generator_config.suffix == "_custom"
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
        generator_function=my_generator_with_resources,
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
        generator_function=my_generator_with_llm,
        input_columns=["input"],
        model_aliases=["test-model"],
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
        generator_function=my_generator,
        input_columns=["input"],
        output_columns=["secondary"],
    )

    row = {"input": 1}

    with pytest.raises(CustomColumnGenerationError, match="did not create declared output_columns"):
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
        generator_function=my_generator,
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
    assert "output_columns" in caplog.text


def test_declared_side_effect_columns_kept() -> None:
    """Test that declared side_effect_columns are kept in the result."""

    def my_generator(row: dict) -> dict:
        row["primary"] = row["input"] * 2
        row["secondary"] = row["input"] * 3
        row["tertiary"] = row["input"] * 4
        return row

    generator = _create_test_generator(
        name="primary",
        generator_function=my_generator,
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


def test_removing_pre_existing_columns_raises_error() -> None:
    """Test that removing pre-existing columns raises an error."""

    def my_generator(row: dict) -> dict:
        row["result"] = row["input"] * 2
        del row["input"]  # Remove a pre-existing column
        return row

    generator = _create_test_generator(
        name="result",
        generator_function=my_generator,
        input_columns=["input"],
    )

    row = {"input": 5, "other": "keep"}

    with pytest.raises(CustomColumnGenerationError, match="removed pre-existing columns"):
        generator.generate(row)


def test_full_column_strategy() -> None:
    """Test that full_column strategy processes DataFrame instead of dict."""
    import pandas as pd

    def batch_processor(df: pd.DataFrame) -> pd.DataFrame:
        df["result"] = df["input"] * 2
        return df

    config = CustomColumnConfig(
        name="result",
        generator_function=batch_processor,
        input_columns=["input"],
        generation_strategy="full_column",
    )
    generator = CustomColumnGenerator(config=config, resource_provider=Mock(spec=ResourceProvider))

    # Verify strategy is FULL_COLUMN
    assert generator.get_generation_strategy() == GenerationStrategy.FULL_COLUMN

    # Test with DataFrame
    df = pd.DataFrame({"input": [1, 2, 3]})
    result = generator.generate(df)

    assert isinstance(result, pd.DataFrame)
    assert "result" in result.columns
    assert list(result["result"]) == [2, 4, 6]


def test_full_column_with_context() -> None:
    """Test full_column strategy with context access."""
    import pandas as pd

    def batch_with_context(df: pd.DataFrame, ctx: CustomColumnContext) -> pd.DataFrame:
        multiplier = ctx.generator_config.multiplier if ctx.generator_config else 1
        df["result"] = df["input"] * multiplier
        return df

    mock_resource_provider = Mock(spec=ResourceProvider)

    config = CustomColumnConfig(
        name="result",
        generator_function=batch_with_context,
        input_columns=["input"],
        generation_strategy="full_column",
        generator_config=SampleGeneratorConfig(multiplier=10),
    )
    generator = CustomColumnGenerator(config=config, resource_provider=mock_resource_provider)

    df = pd.DataFrame({"input": [1, 2, 3]})
    result = generator.generate(df)

    assert list(result["result"]) == [10, 20, 30]


def test_full_column_validates_output() -> None:
    """Test that full_column validates output columns like cell_by_cell."""
    import pandas as pd

    def batch_wrong_column(df: pd.DataFrame) -> pd.DataFrame:
        df["wrong_name"] = df["input"]
        return df

    config = CustomColumnConfig(
        name="expected_name",
        generator_function=batch_wrong_column,
        input_columns=["input"],
        generation_strategy="full_column",
    )
    generator = CustomColumnGenerator(config=config, resource_provider=Mock(spec=ResourceProvider))

    df = pd.DataFrame({"input": [1, 2, 3]})

    with pytest.raises(CustomColumnGenerationError, match="did not create the expected column"):
        generator.generate(df)


def test_full_column_removes_undeclared_columns(caplog: pytest.LogCaptureFixture) -> None:
    """Test that full_column removes undeclared columns with warning."""
    import logging

    import pandas as pd

    def batch_extra_columns(df: pd.DataFrame) -> pd.DataFrame:
        df["result"] = df["input"] * 2
        df["undeclared"] = "should be removed"
        return df

    config = CustomColumnConfig(
        name="result",
        generator_function=batch_extra_columns,
        input_columns=["input"],
        generation_strategy="full_column",
    )
    generator = CustomColumnGenerator(config=config, resource_provider=Mock(spec=ResourceProvider))

    df = pd.DataFrame({"input": [1, 2, 3]})

    with caplog.at_level(logging.WARNING):
        result = generator.generate(df)

    assert "result" in result.columns
    assert "undeclared" not in result.columns
    assert "undeclared columns" in caplog.text


def test_generate_text_batch() -> None:
    """Test that generate_text_batch parallelizes LLM calls."""
    import pandas as pd

    def batch_with_parallel_llm(df: pd.DataFrame, ctx: CustomColumnContext) -> pd.DataFrame:
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

    config = CustomColumnConfig(
        name="result",
        generator_function=batch_with_parallel_llm,
        input_columns=["input"],
        model_aliases=["test-model"],
        generation_strategy="full_column",
    )
    generator = CustomColumnGenerator(config=config, resource_provider=mock_resource_provider)

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
        generator_function=multi_turn_generator,
        input_columns=["topic"],
        model_aliases=["writer", "editor"],
        resource_provider=mock_resource_provider,
    )

    row = {"topic": "testing"}

    # The error from the second LLM call should propagate
    with pytest.raises(CustomColumnGenerationError, match="Custom generator function failed"):
        generator.generate(row)

    # Verify first call succeeded before failure
    assert call_counter["count"] == 2
