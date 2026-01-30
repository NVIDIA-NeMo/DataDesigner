# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import logging

from data_designer.config.column_configs import (
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
)
from data_designer.config.utils.constants import TRACE_COLUMN_POSTFIX
from data_designer.engine.column_generators.generators.base import ColumnGeneratorWithModel, GenerationStrategy
from data_designer.engine.column_generators.utils.prompt_renderer import (
    PromptType,
    RecordBasedPromptRenderer,
    create_response_recipe,
)
from data_designer.engine.configurable_task import TaskConfigT
from data_designer.engine.models.recipes.base import ResponseRecipe
from data_designer.engine.processing.utils import deserialize_json_values

logger = logging.getLogger(__name__)


class ColumnGeneratorWithModelChatCompletion(ColumnGeneratorWithModel[TaskConfigT]):
    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    @functools.cached_property
    def response_recipe(self) -> ResponseRecipe:
        return create_response_recipe(self.config, self.model_config)

    @property
    def max_conversation_correction_steps(self) -> int:
        return self.resource_provider.run_config.max_conversation_correction_steps

    @property
    def max_conversation_restarts(self) -> int:
        return self.resource_provider.run_config.max_conversation_restarts

    @functools.cached_property
    def prompt_renderer(self) -> RecordBasedPromptRenderer:
        return RecordBasedPromptRenderer(
            response_recipe=self.response_recipe,
            error_message_context={
                "column_name": self.config.name,
                "column_type": self.config.column_type,
                "model_alias": self.config.model_alias,
            },
        )

    def generate(self, data: dict) -> dict:
        # Deserialize input data from previous columns so Jinja2 templates can access nested fields
        # Example: If prev column stored '{"key": "value"}', templates can use {{ prev_column.key }}
        # Note: This creates a new dict and doesn't mutate the original `data` argument
        deserialized_record = deserialize_json_values(data)

        multi_modal_context = None
        if self.config.multi_modal_context is not None and len(self.config.multi_modal_context) > 0:
            multi_modal_context = []
            for context in self.config.multi_modal_context:
                multi_modal_context.extend(context.get_contexts(deserialized_record))

        response, trace = self.model.generate(
            prompt=self.prompt_renderer.render(
                record=deserialized_record,
                prompt_template=self.config.prompt,
                prompt_type=PromptType.USER_PROMPT,
            ),
            system_prompt=self.prompt_renderer.render(
                record=deserialized_record,
                prompt_template=self.config.system_prompt,
                prompt_type=PromptType.SYSTEM_PROMPT,
            ),
            parser=self.response_recipe.parse,
            multi_modal_context=multi_modal_context,
            max_correction_steps=self.max_conversation_correction_steps,
            max_conversation_restarts=self.max_conversation_restarts,
            purpose=f"running generation for column '{self.config.name}'",
        )

        serialized_output = self.response_recipe.serialize_output(response)
        data[self.config.name] = self._process_serialized_output(serialized_output)

        should_save_trace = (
            self.config.with_trace or self.resource_provider.run_config.debug_override_save_all_column_traces
        )
        if should_save_trace:
            data[self.config.name + TRACE_COLUMN_POSTFIX] = [message.to_dict() for message in trace]

        return data

    def _process_serialized_output(self, serialized_output: str) -> str | dict | list:
        """Process the serialized output from the model. Subclasses can override to customize deserialization."""
        return serialized_output


class LLMTextCellGenerator(ColumnGeneratorWithModelChatCompletion[LLMTextColumnConfig]): ...


class LLMCodeCellGenerator(ColumnGeneratorWithModelChatCompletion[LLMCodeColumnConfig]): ...


class LLMStructuredCellGenerator(ColumnGeneratorWithModelChatCompletion[LLMStructuredColumnConfig]):
    def _process_serialized_output(self, serialized_output: str) -> dict | list:
        return deserialize_json_values(serialized_output)


class LLMJudgeCellGenerator(ColumnGeneratorWithModelChatCompletion[LLMJudgeColumnConfig]):
    def _process_serialized_output(self, serialized_output: str) -> dict | list:
        return deserialize_json_values(serialized_output)
