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
from data_designer.config.utils.constants import REASONING_CONTENT_COLUMN_POSTFIX, TRACE_COLUMN_POSTFIX
from data_designer.config.utils.trace_type import TraceType
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
            tool_alias=self.config.tool_alias,
            max_correction_steps=self.max_conversation_correction_steps,
            max_conversation_restarts=self.max_conversation_restarts,
            purpose=f"running generation for column '{self.config.name}'",
        )

        serialized_output = self.response_recipe.serialize_output(response)
        data[self.config.name] = self._process_serialized_output(serialized_output)

        effective_trace_type = self.config.with_trace

        if effective_trace_type == TraceType.ALL_MESSAGES:
            data[self.config.name + TRACE_COLUMN_POSTFIX] = [message.to_dict() for message in trace]
        elif effective_trace_type == TraceType.LAST_MESSAGE:
            last_assistant = next((m for m in reversed(trace) if m.role == "assistant"), None)
            data[self.config.name + TRACE_COLUMN_POSTFIX] = [last_assistant.to_dict()] if last_assistant else []

        if self.config.extract_reasoning_content:
            data[self.config.name + REASONING_CONTENT_COLUMN_POSTFIX] = self._extract_reasoning_content(trace)

        return data

    def _extract_reasoning_content(self, trace: list) -> str | None:
        """Extract reasoning_content from the final assistant message in the trace.

        Args:
            trace: List of ChatMessage objects from the generation.

        Returns:
            The stripped reasoning_content from the final assistant message, or None if not present.
        """
        reasoning_value: str | None = None
        for message in reversed(trace):
            if message.role == "assistant":
                reasoning_value = message.reasoning_content
                break

        if reasoning_value is not None:
            reasoning_value = reasoning_value.strip() or None

        return reasoning_value

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
