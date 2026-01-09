# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from data_designer.config.column_configs import (
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
    Score,
)
from data_designer.config.utils.code_lang import CodeLang
from data_designer.engine.column_generators.utils.prompt_renderer import (
    PromptType,
    RecordBasedPromptRenderer,
    create_response_recipe,
)
from data_designer.engine.models.recipes.response_recipes import (
    CodeResponseRecipe,
    PydanticResponseRecipe,
    StructuredResponseRecipe,
    TextResponseRecipe,
)


def test_prompt_renderer_prompt_type():
    assert PromptType.SYSTEM_PROMPT == "system_prompt"
    assert PromptType.USER_PROMPT == "user_prompt"
    assert issubclass(PromptType, str)


@pytest.mark.parametrize(
    "config_class,expected_recipe_class",
    [
        (LLMTextColumnConfig, TextResponseRecipe),
        (LLMCodeColumnConfig, CodeResponseRecipe),
        (LLMStructuredColumnConfig, StructuredResponseRecipe),
        (LLMJudgeColumnConfig, PydanticResponseRecipe),
    ],
)
def test_prompt_renderer_create_response_recipe(config_class, expected_recipe_class):
    if config_class == LLMCodeColumnConfig:
        config = config_class(
            name="test_column", prompt="Test prompt", model_alias="test_model", code_lang=CodeLang.PYTHON
        )
    elif config_class == LLMStructuredColumnConfig:
        config = config_class(
            name="test_column",
            prompt="Test prompt",
            model_alias="test_model",
            output_format={"type": "object", "properties": {"field": {"type": "string"}}},
        )
    elif config_class == LLMJudgeColumnConfig:
        config = config_class(
            name="test_column",
            prompt="Test prompt",
            model_alias="test_model",
            scores=[Score(name="quality", description="Quality score", options={"high": "High", "low": "Low"})],
        )
    else:
        config = config_class(name="test_column", prompt="Test prompt", model_alias="test_model")

    recipe = create_response_recipe(config)

    assert isinstance(recipe, expected_recipe_class)
    if config_class == LLMCodeColumnConfig:
        assert recipe.syntax == CodeLang.PYTHON


def test_prompt_renderer_create_response_recipe_unsupported_type():
    config = Mock()
    config.column_type = "unsupported_type"

    with pytest.raises(ValueError, match="No response recipe found for column type"):
        create_response_recipe(config)


def test_prompt_renderer_record_based_prompt_renderer():
    config = LLMTextColumnConfig(name="test_column", prompt="Test prompt: {{ input }}", model_alias="test_model")

    recipe = create_response_recipe(config)
    renderer = RecordBasedPromptRenderer(response_recipe=recipe)

    assert renderer.response_recipe == recipe


@patch("data_designer.engine.column_generators.utils.prompt_renderer.WithJinja2UserTemplateRendering", autospec=True)
def test_prompt_renderer_render_prompt_success(mock_jinja_mixin):
    config = LLMTextColumnConfig(name="test_column", prompt="Test prompt: {{ input }}", model_alias="test_model")

    recipe = create_response_recipe(config)
    renderer = RecordBasedPromptRenderer(response_recipe=recipe)

    renderer.render = Mock(return_value="Test prompt: Hello World")

    data = {"input": "Hello World"}
    result = renderer.render(
        prompt_template="Test prompt: {{ input }}", record=data, prompt_type=PromptType.USER_PROMPT
    )

    assert result == "Test prompt: Hello World"
    renderer.render.assert_called_once_with(
        prompt_template="Test prompt: {{ input }}", record=data, prompt_type=PromptType.USER_PROMPT
    )


def test_prompt_renderer_render_prompt_template_error():
    config = LLMTextColumnConfig(
        name="test_column", prompt="Test prompt: {{ invalid_template }}", model_alias="test_model"
    )

    recipe = create_response_recipe(config)
    renderer = RecordBasedPromptRenderer(response_recipe=recipe)

    from data_designer.engine.column_generators.utils.errors import PromptTemplateRenderError

    renderer.render = Mock(side_effect=PromptTemplateRenderError("Template error"))

    data = {"input": "Hello World"}

    with pytest.raises(PromptTemplateRenderError, match="Template error"):
        renderer.render(
            prompt_template="Test prompt: {{ invalid_template }}", record=data, prompt_type=PromptType.USER_PROMPT
        )
