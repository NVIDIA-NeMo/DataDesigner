# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from data_designer.config.column_configs import (
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
)
from data_designer.config.run_config import RunConfig
from data_designer.config.utils.constants import TRACE_COLUMN_POSTFIX
from data_designer.engine.column_generators.generators.base import GenerationStrategy
from data_designer.engine.column_generators.generators.llm_completion import (
    LLMCodeCellGenerator,
    LLMJudgeCellGenerator,
    LLMStructuredCellGenerator,
    LLMTextCellGenerator,
)
from data_designer.engine.models.utils import ChatMessage


def _create_generator_with_mocks(config_class=LLMTextColumnConfig, **config_kwargs):
    """Helper function to create generator with mocked dependencies."""
    mock_resource_provider = Mock()
    mock_model_registry = Mock()
    mock_model = Mock()
    mock_model_config = Mock()
    mock_inference_params = Mock()
    mock_prompt_renderer = Mock()
    mock_response_recipe = Mock()
    mock_provider = Mock()

    mock_resource_provider.model_registry = mock_model_registry
    mock_resource_provider.run_config = RunConfig(
        max_conversation_restarts=7,
        max_conversation_correction_steps=2,
    )
    mock_model_registry.get_model.return_value = mock_model
    mock_model_registry.get_model_config.return_value = mock_model_config
    mock_model_registry.get_model_provider.return_value = mock_provider
    mock_model_config.inference_parameters = mock_inference_params
    mock_model_config.alias = "test_model"
    mock_provider.name = "test_provider"

    mock_inference_params.generate_kwargs = {"temperature": 0.7, "max_tokens": 100}

    default_config = {"name": "test_column", "prompt": "Test prompt", "model_alias": "test_model"}
    default_config.update(config_kwargs)

    config = config_class(**default_config)
    generator = LLMTextCellGenerator(config=config, resource_provider=mock_resource_provider)

    generator.prompt_renderer = mock_prompt_renderer
    generator.response_recipe = mock_response_recipe

    return (
        generator,
        mock_resource_provider,
        mock_model,
        mock_model_config,
        mock_inference_params,
        mock_prompt_renderer,
        mock_response_recipe,
    )


def _setup_generate_mocks(mock_prompt_renderer, mock_response_recipe, mock_model, output="test_output"):
    """Helper function to setup common generate method mocks."""
    mock_prompt_renderer.render.side_effect = ["rendered_user_prompt", "rendered_system_prompt"]
    mock_response_recipe.serialize_output.return_value = {"result": output}
    mock_model.generate.return_value = ({"result": output}, [])


def test_generate_method() -> None:
    generator, _, mock_model, _, _, mock_prompt_renderer, mock_response_recipe = _create_generator_with_mocks()

    # Test basic generation
    _setup_generate_mocks(mock_prompt_renderer, mock_response_recipe, mock_model)
    data = {"input": "test_input"}
    result = generator.generate(data)

    assert mock_prompt_renderer.render.call_count == 2
    mock_model.generate.assert_called_once()
    assert mock_model.generate.call_args[1]["max_correction_steps"] == 2
    assert mock_model.generate.call_args[1]["max_conversation_restarts"] == 7
    assert result["test_column"] == {"result": "test_output"}
    assert "test_column" + TRACE_COLUMN_POSTFIX not in result

    # Test with full trace enabled
    mock_model.reset_mock()
    mock_prompt_renderer.reset_mock()
    generator.resource_provider.run_config.debug_override_save_all_column_traces = True
    mock_prompt_renderer.render.side_effect = ["rendered_user_prompt", "rendered_system_prompt"]
    mock_response_recipe.serialize_output.return_value = {"result": "test_output"}
    mock_model.generate.return_value = ({"result": "test_output"}, [ChatMessage.as_user("x")])
    result = generator.generate(data)

    assert result["test_column"] == {"result": "test_output"}
    assert result["test_column" + TRACE_COLUMN_POSTFIX] == [{"role": "user", "content": "x"}]

    # Test multi-modal context is None
    call_args = mock_model.generate.call_args
    assert call_args[1]["multi_modal_context"] is None


@patch("data_designer.engine.column_generators.generators.base.logger", autospec=True)
def test_log_pre_generation(mock_logger: Mock) -> None:
    generator, mock_resource_provider, _, mock_model_config, mock_inference_params, _, _ = (
        _create_generator_with_mocks()
    )
    mock_model_config.model = "meta/llama-3.1-8b-instruct"
    mock_model_config.generation_type.value = "chat-completion"
    mock_inference_params.format_for_display.return_value = "temperature=0.70, max_tokens=100"

    generator.log_pre_generation()

    assert mock_logger.info.call_count == 5
    mock_logger.info.assert_any_call("📝 llm-text model config for column 'test_column'")
    mock_logger.info.assert_any_call("  |-- model: 'meta/llama-3.1-8b-instruct'")
    mock_logger.info.assert_any_call("  |-- model alias: 'test_model'")
    mock_logger.info.assert_any_call("  |-- model provider: 'test_provider'")
    mock_logger.info.assert_any_call("  |-- inference parameters: temperature=0.70, max_tokens=100")

    # Test with different provider
    mock_logger.reset_mock()
    mock_provider = Mock()
    mock_provider.name = "test_provider_2"
    mock_resource_provider.model_registry.get_model_provider.return_value = mock_provider

    generator.log_pre_generation()
    mock_logger.info.assert_any_call("  |-- model provider: 'test_provider_2'")


@pytest.mark.parametrize(
    "generator_class",
    [
        LLMTextCellGenerator,
        LLMCodeCellGenerator,
        LLMJudgeCellGenerator,
        LLMStructuredCellGenerator,
    ],
)
def test_llm_generator_generation_strategy(generator_class: type) -> None:
    assert generator_class.get_generation_strategy() == GenerationStrategy.CELL_BY_CELL


@pytest.mark.parametrize(
    "generator_class,config_class,config_kwargs",
    [
        (
            LLMTextCellGenerator,
            LLMTextColumnConfig,
            {"name": "test_column", "prompt": "Generate text: {{ input }}", "model_alias": "test_model"},
        ),
        (
            LLMCodeCellGenerator,
            LLMCodeColumnConfig,
            {
                "name": "test_column",
                "prompt": "Generate code: {{ input }}",
                "model_alias": "test_model",
                "code_lang": "python",
            },
        ),
        (
            LLMJudgeCellGenerator,
            LLMJudgeColumnConfig,
            {
                "name": "test_column",
                "prompt": "Judge: {{ input }}",
                "model_alias": "test_model",
                "scores": [{"name": "quality", "description": "Quality", "options": {1: "good", 0: "bad"}}],
            },
        ),
        (
            LLMStructuredCellGenerator,
            LLMStructuredColumnConfig,
            {
                "name": "test_column",
                "prompt": "Generate structured data: {{ input }}",
                "model_alias": "test_model",
                "output_format": {"type": "object", "properties": {"field": {"type": "string"}}},
            },
        ),
    ],
)
def test_llm_generator_creation(generator_class, config_class, config_kwargs):
    config = config_class(**config_kwargs)
    mock_resource_provider = Mock()
    generator = generator_class(config=config, resource_provider=mock_resource_provider)
    assert generator.config == config


def test_judge_generator_max_conversation_restarts_override():
    mock_resource_provider = Mock()
    mock_model_registry = Mock()
    mock_model = Mock()
    mock_model_config = Mock()
    mock_inference_params = Mock()

    mock_resource_provider.model_registry = mock_model_registry
    mock_resource_provider.run_config = RunConfig(
        max_conversation_restarts=7,
        max_conversation_correction_steps=2,
    )
    mock_model_registry.get_model.return_value = mock_model
    mock_model_registry.get_model_config.return_value = mock_model_config
    mock_model_config.inference_parameters = mock_inference_params

    config = LLMJudgeColumnConfig(
        name="test_column",
        prompt="Judge this: {{ input }}",
        model_alias="test_model",
        scores=[{"name": "quality", "description": "Quality", "options": {1: "good", 0: "bad"}}],
    )

    generator = LLMJudgeCellGenerator(config=config, resource_provider=mock_resource_provider)

    assert generator.max_conversation_restarts == 7
    assert generator.max_conversation_correction_steps == 2


@pytest.mark.parametrize(
    "error_type,error_message",
    [
        ("serialization", "Serialization error"),
        ("model", "Model generation error"),
        ("prompt_render", "Prompt render error"),
    ],
)
def test_generate_with_errors(error_type, error_message):
    generator, _, mock_model, _, _, mock_prompt_renderer, mock_response_recipe = _create_generator_with_mocks()

    mock_prompt_renderer.render.side_effect = ["rendered_user_prompt", "rendered_system_prompt"]

    if error_type == "serialization":
        mock_response_recipe.serialize_output.side_effect = Exception(error_message)
        mock_model.generate.return_value = ({"result": "test_output"}, [])
    elif error_type == "model":
        mock_model.generate.side_effect = Exception(error_message)
    elif error_type == "prompt_render":
        mock_prompt_renderer.render.side_effect = Exception(error_message)

    data = {"input": "test_input"}

    with pytest.raises(Exception, match=error_message):
        generator.generate(data)


def test_generate_with_complex_data():
    generator, _, mock_model, _, _, mock_prompt_renderer, mock_response_recipe = _create_generator_with_mocks()
    _setup_generate_mocks(mock_prompt_renderer, mock_response_recipe, mock_model, "complex_output")

    data = {"input": "test_input", "nested": {"key": "value"}, "list": [1, 2, 3], "json_string": '{"key": "value"}'}
    result = generator.generate(data)

    assert result["test_column"] == {"result": "complex_output"}
    assert result["input"] == "test_input"
    assert result["nested"] == {"key": "value"}
    assert result["list"] == [1, 2, 3]


def test_generate_with_json_deserialization():
    generator, _, mock_model, _, _, mock_prompt_renderer, mock_response_recipe = _create_generator_with_mocks()
    _setup_generate_mocks(mock_prompt_renderer, mock_response_recipe, mock_model, "json_output")

    data = {"json_field": '{"nested": {"value": 123}}', "string_field": "regular_string"}
    result = generator.generate(data)

    assert result["test_column"] == {"result": "json_output"}


@pytest.mark.parametrize(
    "generator_class,config_class,config_kwargs,serialized_output,expected_output",
    [
        (
            LLMTextCellGenerator,
            LLMTextColumnConfig,
            {"name": "text_col", "prompt": "Generate text", "model_alias": "test_model"},
            '["plain", "text", "output"]',
            '["plain", "text", "output"]',
        ),
        (
            LLMCodeCellGenerator,
            LLMCodeColumnConfig,
            {"name": "code_col", "prompt": "Generate code", "model_alias": "test_model", "code_lang": "python"},
            "def hello(): pass",
            "def hello(): pass",
        ),
        (
            LLMStructuredCellGenerator,
            LLMStructuredColumnConfig,
            {
                "name": "struct_col",
                "prompt": "Generate struct",
                "model_alias": "test_model",
                "output_format": {"type": "object", "properties": {"field": {"type": "string"}}},
            },
            '{"field": "value", "nested": {"key": "val"}}',
            {"field": "value", "nested": {"key": "val"}},
        ),
        (
            LLMJudgeCellGenerator,
            LLMJudgeColumnConfig,
            {
                "name": "judge_col",
                "prompt": "Judge this",
                "model_alias": "test_model",
                "scores": [{"name": "quality", "description": "Quality", "options": {1: "good", 0: "bad"}}],
            },
            '{"quality": 1, "reasoning": "Good quality"}',
            {"quality": 1, "reasoning": "Good quality"},
        ),
    ],
)
def test_generator_output_type_handling(
    stub_resource_provider: Mock,
    generator_class: type,
    config_class: type,
    config_kwargs: dict,
    serialized_output: str,
    expected_output: str | dict,
) -> None:
    """Test that each generator type correctly handles its output format via polymorphism.

    - Text/Code generators return plain strings
    - Structured/Judge generators deserialize JSON to Python objects
    """
    config = config_class(**config_kwargs)
    generator = generator_class(config=config, resource_provider=stub_resource_provider)

    # Mock the prompt renderer and response recipe
    mock_prompt_renderer = Mock()
    mock_response_recipe = Mock()
    generator.prompt_renderer = mock_prompt_renderer
    generator.response_recipe = mock_response_recipe

    # Setup mocks
    mock_prompt_renderer.render.side_effect = ["rendered_user_prompt", "rendered_system_prompt"]
    mock_response_recipe.serialize_output.return_value = serialized_output
    stub_resource_provider.model_registry.get_model.return_value.generate.return_value = (
        {"result": "raw_output"},
        [],
    )

    data = {"input": "test_input"}
    result = generator.generate(data)

    assert result[config.name] == expected_output
