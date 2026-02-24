# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import functools
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, overload

from data_designer.config.column_configs import GenerationStrategy
from data_designer.engine.configurable_task import ConfigurableTask, DataT, TaskConfigT
from data_designer.logging import LOG_DOUBLE_INDENT, LOG_INDENT

if TYPE_CHECKING:
    import pandas as pd

    from data_designer.config.models import BaseInferenceParams, ModelConfig
    from data_designer.engine.models.facade import ModelFacade
    from data_designer.engine.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class ColumnGenerator(ConfigurableTask[TaskConfigT], ABC):
    @property
    def can_generate_from_scratch(self) -> bool:
        return False

    @staticmethod
    @abstractmethod
    def get_generation_strategy() -> GenerationStrategy: ...

    @overload
    @abstractmethod
    def generate(self, data: dict) -> dict: ...

    @overload
    @abstractmethod
    def generate(self, data: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def generate(self, data: DataT) -> DataT: ...

    async def agenerate(self, data: dict) -> dict:
        """Async fallback — delegates to sync generate via thread pool.

        Subclasses with native async support (e.g. ColumnGeneratorWithModelChatCompletion)
        should override this with a direct async implementation.
        """
        return await asyncio.to_thread(self.generate, data)

    def log_pre_generation(self) -> None:
        """A shared method to log info before the generator's `generate` method is called.

        The idea is for dataset builders to call this method for all generators before calling their
        `generate` method. This is to avoid logging the same information multiple times when running
        generators in parallel.
        """


class FromScratchColumnGenerator(ColumnGenerator[TaskConfigT], ABC):
    @property
    def can_generate_from_scratch(self) -> bool:
        return True

    @abstractmethod
    def generate_from_scratch(self, num_records: int) -> pd.DataFrame: ...


class ColumnGeneratorWithModelRegistry(ColumnGenerator[TaskConfigT], ABC):
    @property
    def model_registry(self) -> ModelRegistry:
        return self.resource_provider.model_registry

    def get_model(self, model_alias: str) -> ModelFacade:
        return self.model_registry.get_model(model_alias=model_alias)

    def get_model_config(self, model_alias: str) -> ModelConfig:
        return self.model_registry.get_model_config(model_alias=model_alias)

    def get_model_provider_name(self, model_alias: str) -> str:
        provider = self.model_registry.get_model_provider(model_alias=model_alias)
        return provider.name


class ColumnGeneratorWithModel(ColumnGeneratorWithModelRegistry[TaskConfigT], ABC):
    @functools.cached_property
    def model(self) -> ModelFacade:
        return self.get_model(model_alias=self.config.model_alias)

    @functools.cached_property
    def model_config(self) -> ModelConfig:
        return self.get_model_config(model_alias=self.config.model_alias)

    @functools.cached_property
    def inference_parameters(self) -> BaseInferenceParams:
        return self.model_config.inference_parameters

    def _build_multi_modal_context(self, record: dict) -> list[dict[str, Any]] | None:
        """Build multi-modal context from the config's multi_modal_context list.

        Passes base_path to get_contexts() so that generated image file paths
        (stored under base_dataset_path in create mode) can be resolved to base64
        before being sent to the model endpoint.

        Args:
            record: The deserialized record containing column values.

        Returns:
            A list of multi-modal context dicts, or None if no context is configured.
        """
        if not hasattr(self.config, "multi_modal_context") or not self.config.multi_modal_context:
            return None
        base_path = str(self.base_dataset_path)
        multi_modal_context: list[dict[str, Any]] = []
        for context in self.config.multi_modal_context:
            multi_modal_context.extend(context.get_contexts(record, base_path=base_path))
        return multi_modal_context

    def log_pre_generation(self) -> None:
        logger.info(
            f"{self.config.get_column_emoji()} {self.config.column_type} model config for column '{self.config.name}'"
        )
        logger.info(f"{LOG_INDENT}model: {self.model_config.model!r}")
        logger.info(f"{LOG_INDENT}model alias: {self.config.model_alias!r}")
        logger.info(
            f"{LOG_INDENT}model provider: {self.get_model_provider_name(model_alias=self.config.model_alias)!r}"
        )
        logger.info(f"{LOG_INDENT}inference parameters:")
        for param in self.inference_parameters.get_formatted_params():
            logger.info(f"{LOG_DOUBLE_INDENT}{param}")

        tool_alias = getattr(self.config, "tool_alias", None)
        if tool_alias is not None:
            tool_config = self.resource_provider.mcp_registry.get_tool_config(tool_alias=tool_alias)
            logger.info(f"{LOG_INDENT}tool alias: {tool_alias!r}")
            logger.info(f"{LOG_INDENT}mcp providers: {tool_config.providers!r}")


class ColumnGeneratorCellByCell(ColumnGenerator[TaskConfigT], ABC):
    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    @abstractmethod
    def generate(self, data: dict) -> dict: ...


class ColumnGeneratorFullColumn(ColumnGenerator[TaskConfigT], ABC):
    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.FULL_COLUMN

    @abstractmethod
    def generate(self, data: pd.DataFrame) -> pd.DataFrame: ...
