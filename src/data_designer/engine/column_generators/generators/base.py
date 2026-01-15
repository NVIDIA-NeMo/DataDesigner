# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, overload

from data_designer.engine.configurable_task import ConfigurableTask, DataT, TaskConfigT
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd

    from data_designer.config.models import BaseInferenceParams, ModelConfig
    from data_designer.engine.models.facade import ModelFacade
    from data_designer.engine.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class GenerationStrategy(str, Enum):
    CELL_BY_CELL = "cell_by_cell"
    FULL_COLUMN = "full_column"


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

    def log_pre_generation(self) -> None:
        logger.info(
            f"{self.config.get_column_emoji()} {self.config.column_type} model config for column '{self.config.name}'"
        )
        logger.info(f"  |-- model: {self.model_config.model!r}")
        logger.info(f"  |-- model alias: {self.config.model_alias!r}")
        logger.info(f"  |-- model provider: {self.get_model_provider_name(model_alias=self.config.model_alias)!r}")
        logger.info(f"  |-- inference parameters: {self.inference_parameters.format_for_display()}")


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
