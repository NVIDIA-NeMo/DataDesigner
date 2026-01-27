# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.errors import InvalidConfigError
from data_designer.engine.column_generators.generators.custom import CustomColumnGenerator
from data_designer.engine.column_generators.utils.errors import CustomColumnGenerationError
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd


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
        def generate_fn(df: pd.DataFrame) -> pd.DataFrame:
            df[name] = "test_value"
            return df

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


def test_generate_simple_column() -> None:
    """Test generating a simple column."""
    def my_generator(df: pd.DataFrame) -> pd.DataFrame:
        df["result"] = df["input"].apply(lambda x: x.upper())
        return df

    generator = _create_test_generator(
        name="result",
        generate_fn=my_generator,
        input_columns=["input"],
    )

    df = pd.DataFrame({"input": ["hello", "world"]})
    result = generator.generate(df)

    assert "result" in result.columns
    assert result["result"].tolist() == ["HELLO", "WORLD"]


def test_generate_with_kwargs() -> None:
    """Test that kwargs are accessible via config."""
    def my_generator(df: pd.DataFrame) -> pd.DataFrame:
        # In a real scenario, the generator would access kwargs differently
        # For this test, we just verify the config has the kwargs
        df["result"] = "processed"
        return df

    generator = _create_test_generator(
        name="result",
        generate_fn=my_generator,
        kwargs={"multiplier": 2, "prefix": "test_"},
    )

    assert generator.config.kwargs == {"multiplier": 2, "prefix": "test_"}


def test_generate_missing_required_columns() -> None:
    """Test that missing required columns raise an error."""
    def my_generator(df: pd.DataFrame) -> pd.DataFrame:
        df["result"] = df["missing_column"]
        return df

    generator = _create_test_generator(
        name="result",
        generate_fn=my_generator,
        input_columns=["missing_column"],
    )

    df = pd.DataFrame({"other_column": [1, 2, 3]})

    with pytest.raises(CustomColumnGenerationError, match="Missing required columns"):
        generator.generate(df)


def test_generate_function_raises_error() -> None:
    """Test that errors in the generate function are wrapped."""
    def my_generator(df: pd.DataFrame) -> pd.DataFrame:
        raise ValueError("Something went wrong")

    generator = _create_test_generator(
        name="result",
        generate_fn=my_generator,
    )

    df = pd.DataFrame({"input": [1, 2, 3]})

    with pytest.raises(CustomColumnGenerationError, match="Custom generator function failed"):
        generator.generate(df)


def test_generate_returns_non_dataframe() -> None:
    """Test that returning non-DataFrame raises an error."""
    def my_generator(df: pd.DataFrame) -> list:
        return [1, 2, 3]

    generator = _create_test_generator(
        name="result",
        generate_fn=my_generator,
    )

    df = pd.DataFrame({"input": [1, 2, 3]})

    with pytest.raises(CustomColumnGenerationError, match="must return a pandas DataFrame"):
        generator.generate(df)


def test_generate_missing_output_column() -> None:
    """Test that not creating the expected column raises an error."""
    def my_generator(df: pd.DataFrame) -> pd.DataFrame:
        df["wrong_column"] = "value"
        return df

    generator = _create_test_generator(
        name="expected_column",
        generate_fn=my_generator,
    )

    df = pd.DataFrame({"input": [1, 2, 3]})

    with pytest.raises(CustomColumnGenerationError, match="did not create the expected column"):
        generator.generate(df)


def test_config_validation_non_callable() -> None:
    """Test that non-callable generate_fn raises an error."""
    with pytest.raises(InvalidConfigError, match="must be a callable"):
        CustomColumnConfig(
            name="test",
            generate_fn="not_a_function",
        )


def test_config_required_columns_property() -> None:
    """Test that required_columns returns input_columns."""
    def my_generator(df: pd.DataFrame) -> pd.DataFrame:
        return df

    config = CustomColumnConfig(
        name="test",
        generate_fn=my_generator,
        input_columns=["col1", "col2"],
    )

    assert config.required_columns == ["col1", "col2"]


def test_config_side_effect_columns_property() -> None:
    """Test that side_effect_columns returns output_columns."""
    def my_generator(df: pd.DataFrame) -> pd.DataFrame:
        return df

    config = CustomColumnConfig(
        name="test",
        generate_fn=my_generator,
        output_columns=["extra_col1", "extra_col2"],
    )

    assert config.side_effect_columns == ["extra_col1", "extra_col2"]


def test_generate_multiple_output_columns() -> None:
    """Test generating multiple output columns."""
    def my_generator(df: pd.DataFrame) -> pd.DataFrame:
        df["primary"] = df["input"] * 2
        df["secondary"] = df["input"] * 3
        return df

    generator = _create_test_generator(
        name="primary",
        generate_fn=my_generator,
        input_columns=["input"],
        output_columns=["secondary"],
    )

    df = pd.DataFrame({"input": [1, 2, 3]})
    result = generator.generate(df)

    assert "primary" in result.columns
    assert "secondary" in result.columns
    assert result["primary"].tolist() == [2, 4, 6]
    assert result["secondary"].tolist() == [3, 6, 9]


def test_log_pre_generation(caplog: pytest.LogCaptureFixture) -> None:
    """Test that log_pre_generation logs correctly."""
    import logging

    def my_generator(df: pd.DataFrame) -> pd.DataFrame:
        df["result"] = "value"
        return df

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
    def my_custom_function(df: pd.DataFrame) -> pd.DataFrame:
        return df

    config = CustomColumnConfig(
        name="test",
        generate_fn=my_custom_function,
    )

    serialized = config.model_dump()
    assert serialized["generate_fn"] == "my_custom_function"


def test_generate_with_context_access() -> None:
    """Test that a two-argument function receives a CustomColumnContext."""
    from data_designer.config.custom_column import CustomColumnContext

    received_context = None

    def my_generator_with_access(
        df: pd.DataFrame, ctx: CustomColumnContext
    ) -> pd.DataFrame:
        nonlocal received_context
        received_context = ctx
        # Access kwargs through the context
        suffix = ctx.kwargs.get("suffix", "_processed")
        df["result"] = df["input"].apply(lambda x: f"{x}{suffix}")
        return df

    generator = _create_test_generator(
        name="result",
        generate_fn=my_generator_with_access,
        input_columns=["input"],
        kwargs={"suffix": "_custom"},
    )

    df = pd.DataFrame({"input": ["a", "b", "c"]})
    result = generator.generate(df)

    # Verify the context was passed and has correct properties
    assert received_context is not None
    assert isinstance(received_context, CustomColumnContext)
    assert received_context.kwargs == {"suffix": "_custom"}
    assert received_context.column_name == "result"

    # Verify the result
    assert result["result"].tolist() == ["a_custom", "b_custom", "c_custom"]


def test_context_provides_model_registry_access() -> None:
    """Test that CustomColumnContext provides access to model_registry."""
    from data_designer.config.custom_column import CustomColumnContext

    accessed_model_registry = None

    def my_generator_with_resources(
        df: pd.DataFrame, ctx: CustomColumnContext
    ) -> pd.DataFrame:
        nonlocal accessed_model_registry
        accessed_model_registry = ctx.model_registry
        df["result"] = "processed"
        return df

    mock_resource_provider = Mock(spec=ResourceProvider)
    mock_model_registry = Mock()
    mock_resource_provider.model_registry = mock_model_registry

    generator = _create_test_generator(
        name="result",
        generate_fn=my_generator_with_resources,
        resource_provider=mock_resource_provider,
    )

    df = pd.DataFrame({"input": [1, 2, 3]})
    generator.generate(df)

    # Verify the model_registry was accessible via context
    assert accessed_model_registry is mock_model_registry


def test_context_generate_text() -> None:
    """Test that CustomColumnContext.generate_text calls the model correctly."""
    from data_designer.config.custom_column import CustomColumnContext

    generated_texts = []

    def my_generator_with_llm(
        df: pd.DataFrame, ctx: CustomColumnContext
    ) -> pd.DataFrame:
        for _, row in df.iterrows():
            text = ctx.generate_text(
                model_alias="test-model",
                prompt=f"Process: {row['input']}",
                system_prompt="You are helpful.",
            )
            generated_texts.append(text)
        df["result"] = generated_texts
        return df

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

    df = pd.DataFrame({"input": ["a", "b"]})
    result = generator.generate(df)

    # Verify the model was called correctly
    assert mock_model_registry.get_model.call_count == 2
    mock_model_registry.get_model.assert_called_with(model_alias="test-model")

    # Verify generate was called with correct parameters
    assert mock_model.generate.call_count == 2
    call_kwargs = mock_model.generate.call_args_list[0][1]
    assert "Process: a" in call_kwargs["prompt"]
    assert call_kwargs["system_prompt"] == "You are helpful."

    # Verify results
    assert result["result"].tolist() == ["Generated text", "Generated text"]


def test_context_generate_text_batch() -> None:
    """Test that CustomColumnContext.generate_text_batch parallelizes calls."""
    from data_designer.config.custom_column import CustomColumnContext

    def my_batch_generator(
        df: pd.DataFrame, ctx: CustomColumnContext
    ) -> pd.DataFrame:
        prompts = [f"Process: {row['input']}" for _, row in df.iterrows()]
        results = ctx.generate_text_batch(
            model_alias="test-model",
            prompts=prompts,
            system_prompt="You are helpful.",
            max_workers=2,
        )
        df["result"] = results
        return df

    # Set up mocks
    mock_resource_provider = Mock(spec=ResourceProvider)
    mock_model_registry = Mock()
    mock_model_config = Mock()
    mock_model_config.inference_parameters = Mock()
    mock_model_config.inference_parameters.max_parallel_requests = 4
    mock_model = Mock()

    # Return different responses for each call
    call_count = [0]

    def mock_generate(**kwargs):
        call_count[0] += 1
        return (f"Response {call_count[0]}", None)

    mock_model.generate.side_effect = mock_generate
    mock_model_registry.get_model.return_value = mock_model
    mock_model_registry.get_model_config.return_value = mock_model_config
    mock_resource_provider.model_registry = mock_model_registry

    generator = _create_test_generator(
        name="result",
        generate_fn=my_batch_generator,
        input_columns=["input"],
        resource_provider=mock_resource_provider,
    )

    df = pd.DataFrame({"input": ["a", "b", "c"]})
    result = generator.generate(df)

    # Verify the model was called 3 times (once per prompt)
    assert mock_model.generate.call_count == 3

    # Verify results are in correct order (indices preserved)
    assert len(result["result"]) == 3
    # Note: order may vary due to parallelism, but all should be present
    assert all("Response" in str(r) for r in result["result"])


def test_context_generate_text_batch_empty() -> None:
    """Test that generate_text_batch handles empty prompts."""
    from data_designer.config.custom_column import CustomColumnContext

    mock_resource_provider = Mock(spec=ResourceProvider)
    mock_config = Mock()
    mock_config.kwargs = {}
    mock_config.name = "test"

    ctx = CustomColumnContext(
        resource_provider=mock_resource_provider,
        config=mock_config,
    )

    # Empty list should return empty list
    results = ctx.generate_text_batch(
        model_alias="test-model",
        prompts=[],
    )
    assert results == []
