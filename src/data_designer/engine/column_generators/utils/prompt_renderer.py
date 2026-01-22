# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging

from data_designer.config.column_configs import SingleColumnConfig
from data_designer.config.column_types import DataDesignerColumnType
from data_designer.config.models import ModelConfig
from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.utils.misc import extract_keywords_from_jinja2_template
from data_designer.config.utils.type_helpers import StrEnum
from data_designer.engine.column_generators.utils.errors import PromptTemplateRenderError
from data_designer.engine.column_generators.utils.judge_score_factory import (
    create_judge_response_model,
    create_judge_structured_output_model,
)
from data_designer.engine.models.recipes.base import ResponseRecipe
from data_designer.engine.models.recipes.response_recipes import (
    CodeResponseRecipe,
    PydanticResponseRecipe,
    StructuredResponseRecipe,
    TextResponseRecipe,
)
from data_designer.engine.processing.ginja.environment import WithJinja2UserTemplateRendering
from data_designer.engine.processing.ginja.exceptions import UserTemplateError, UserTemplateUnsupportedFiltersError

logger = logging.getLogger(__name__)


class PromptType(StrEnum):
    SYSTEM_PROMPT = "system_prompt"
    USER_PROMPT = "user_prompt"


class RecordBasedPromptRenderer(WithJinja2UserTemplateRendering):
    def __init__(self, response_recipe: ResponseRecipe, *, error_message_context: dict[str, str] | None = None):
        self.response_recipe = response_recipe
        self._error_message_context = error_message_context

    def render(self, *, prompt_template: str | None, record: dict, prompt_type: PromptType) -> str | None:
        self._prepare_environment(prompt_template=prompt_template, record=record, prompt_type=prompt_type)
        rendered_prompt = self.render_multi_template(prompt_type, record) if prompt_template else ""
        recipe_applicator = (
            self.response_recipe.apply_recipe_to_user_prompt
            if prompt_type == PromptType.USER_PROMPT
            else self.response_recipe.apply_recipe_to_system_prompt
        )
        return recipe_applicator(rendered_prompt)

    def _prepare_environment(self, *, prompt_template: str | None, record: dict, prompt_type: PromptType) -> None:
        try:
            self.prepare_jinja2_multi_template_renderer(
                template_name=prompt_type.value,
                prompt_template=prompt_template,
                dataset_variables=list(record.keys()),
            )
        except (UserTemplateUnsupportedFiltersError, UserTemplateError) as exc:
            template_variables = extract_keywords_from_jinja2_template(prompt_template)
            missing_columns = list(set(template_variables) - set(record.keys()))

            error_msg = (
                f"There was an error preparing the {prompt_type.value.replace('_', ' ')} "
                "template. Please double check that the template is valid Jinja2 syntax, that all "
                "referenced variables are defined, and that any filters you are using are supported."
            )
            if len(missing_columns) > 0:
                error_msg += f"\nThe following {missing_columns} columns are missing!"
            if self._error_message_context is not None:
                error_msg += f"\n{json.dumps(self._error_message_context, indent=2)}"
            logger.error(f"🛑 {error_msg}")
            raise PromptTemplateRenderError(f"{exc!s} {error_msg}")


def create_response_recipe(
    column_config: SingleColumnConfig, model_config: ModelConfig | None = None
) -> ResponseRecipe:
    if model_config and column_config.model_alias != model_config.alias:
        raise ValueError(
            f"Column config model alias {column_config.model_alias} does not match model config alias {model_config.alias}"
        )
    if column_config.column_type == DataDesignerColumnType.LLM_TEXT:
        return TextResponseRecipe()
    if column_config.column_type == DataDesignerColumnType.LLM_CODE:
        return CodeResponseRecipe(
            syntax=CodeLang.parse_lang(column_config.code_lang),
        )
    if column_config.column_type == DataDesignerColumnType.LLM_STRUCTURED:
        return StructuredResponseRecipe(
            json_schema=column_config.output_format,
        )
    if column_config.column_type == DataDesignerColumnType.LLM_JUDGE:
        return PydanticResponseRecipe(
            data_type=create_judge_structured_output_model(
                [create_judge_response_model(s) for s in column_config.scores]
            ),
        )
    raise ValueError(f"No response recipe found for column type: {column_config.column_type}")
