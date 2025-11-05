# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import logging

from data_designer.config.columns import (
    COLUMN_TYPE_EMOJI_MAP,
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
)
from data_designer.config.models import InferenceParameters, ModelConfig
from data_designer.config.utils.constants import REASONING_TRACE_COLUMN_POSTFIX
from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    GenerationStrategy,
    GeneratorMetadata,
)
from data_designer.engine.column_generators.utils.prompt_renderer import (
    PromptType,
    RecordBasedPromptRenderer,
    create_response_recipe,
)
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.models.recipes.base import ResponseRecipe
from data_designer.engine.processing.utils import deserialize_json_values
from data_designer.engine.resources.resource_provider import ResourceType

DEFAULT_MAX_CONVERSATION_RESTARTS = 5
DEFAULT_MAX_CONVERSATION_CORRECTION_STEPS = 0


logger = logging.getLogger(__name__)


class WithLLMGeneration:
    @functools.cached_property
    def model(self) -> ModelFacade:
        return self.resource_provider.model_registry.get_model(model_alias=self.config.model_alias)

    @functools.cached_property
    def model_config(self) -> ModelConfig:
        return self.resource_provider.model_registry.get_model_config(model_alias=self.config.model_alias)

    @functools.cached_property
    def inference_parameters(self) -> InferenceParameters:
        return self.model_config.inference_parameters

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

    @functools.cached_property
    def response_recipe(self) -> ResponseRecipe:
        return create_response_recipe(self.config, self.model_config)

    @property
    def max_conversation_correction_steps(self) -> int:
        return DEFAULT_MAX_CONVERSATION_CORRECTION_STEPS

    @property
    def max_conversation_restarts(self) -> int:
        return DEFAULT_MAX_CONVERSATION_RESTARTS

    def generate(self, data: dict) -> dict:
        deserialized_record = deserialize_json_values(data)

        multi_modal_context = None
        if self.config.multi_modal_context is not None and len(self.config.multi_modal_context) > 0:
            multi_modal_context = [
                context.get_context(deserialized_record) for context in self.config.multi_modal_context
            ]

        response, reasoning_trace = self.model.generate(
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
            **self.inference_parameters.generate_kwargs,
        )

        data[self.config.name] = deserialize_json_values(self.response_recipe.serialize_output(response))

        if reasoning_trace:
            data[self.config.name + REASONING_TRACE_COLUMN_POSTFIX] = reasoning_trace

        return data

    def log_pre_generation(self) -> None:
        emoji = COLUMN_TYPE_EMOJI_MAP[self.config.column_type]
        logger.info(f"{emoji} Preparing {self.config.column_type} column generation")
        logger.info(f"  |-- column name: {self.config.name!r}")
        logger.info(f"  |-- model config:\n{self.model_config.model_dump_json(indent=4)}")
        if self.model_config.provider is None:
            logger.info(f"  |-- default model provider: {self._get_provider_name()!r}")

    def _get_provider_name(self) -> str:
        model_alias = self.model_config.alias
        provider = self.resource_provider.model_registry.get_model_provider(model_alias=model_alias)
        return provider.name


class LLMTextCellGenerator(WithLLMGeneration, ColumnGenerator[LLMTextColumnConfig]):
    @staticmethod
    def metadata() -> GeneratorMetadata:
        return GeneratorMetadata(
            name="llm_text_generator",
            description="Generate a new dataset cell from a prompt template",
            generation_strategy=GenerationStrategy.CELL_BY_CELL,
            required_resources=[ResourceType.MODEL_REGISTRY],
        )


class LLMCodeCellGenerator(WithLLMGeneration, ColumnGenerator[LLMCodeColumnConfig]):
    @staticmethod
    def metadata() -> GeneratorMetadata:
        return GeneratorMetadata(
            name="llm_code_generator",
            description="Generate a new dataset cell from a prompt template",
            generation_strategy=GenerationStrategy.CELL_BY_CELL,
            required_resources=[ResourceType.MODEL_REGISTRY],
        )


class LLMStructuredCellGenerator(WithLLMGeneration, ColumnGenerator[LLMStructuredColumnConfig]):
    @staticmethod
    def metadata() -> GeneratorMetadata:
        return GeneratorMetadata(
            name="llm_structured_generator",
            description="Generate a new dataset cell from a prompt template",
            generation_strategy=GenerationStrategy.CELL_BY_CELL,
            required_resources=[ResourceType.MODEL_REGISTRY],
        )


class LLMJudgeCellGenerator(WithLLMGeneration, ColumnGenerator[LLMJudgeColumnConfig]):
    @staticmethod
    def metadata() -> GeneratorMetadata:
        return GeneratorMetadata(
            name="llm_judge_generator",
            description="Judge a new dataset cell based on a set of rubrics",
            generation_strategy=GenerationStrategy.CELL_BY_CELL,
            required_resources=[ResourceType.MODEL_REGISTRY],
        )

    @property
    def max_conversation_correction_steps(self) -> int:
        return DEFAULT_MAX_CONVERSATION_CORRECTION_STEPS

    @property
    def max_conversation_restarts(self) -> int:
        return 2 * DEFAULT_MAX_CONVERSATION_RESTARTS
