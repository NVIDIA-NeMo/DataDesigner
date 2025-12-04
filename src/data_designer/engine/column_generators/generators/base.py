# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import functools
import logging
from typing import overload

import pandas as pd

from data_designer.config.column_types import COLUMN_TYPE_EMOJI_MAP
from data_designer.config.models import BaseInferenceParameters, ModelConfig
from data_designer.config.utils.type_helpers import StrEnum
from data_designer.engine.configurable_task import ConfigurableTask, ConfigurableTaskMetadata, DataT, TaskConfigT
from data_designer.engine.models.facade import ModelFacade

logger = logging.getLogger(__name__)


class GenerationStrategy(StrEnum):
    CELL_BY_CELL = "cell_by_cell"
    FULL_COLUMN = "full_column"


class GeneratorMetadata(ConfigurableTaskMetadata):
    generation_strategy: GenerationStrategy


class ColumnGenerator(ConfigurableTask[TaskConfigT], ABC):
    @property
    def can_generate_from_scratch(self) -> bool:
        return False

    @property
    def generation_strategy(self) -> GenerationStrategy:
        return self.metadata().generation_strategy

    @staticmethod
    @abstractmethod
    def metadata() -> GeneratorMetadata: ...

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


class WithModelGeneration:
    @functools.cached_property
    def model(self) -> ModelFacade:
        return self.resource_provider.model_registry.get_model(model_alias=self.config.model_alias)

    @functools.cached_property
    def model_config(self) -> ModelConfig:
        return self.resource_provider.model_registry.get_model_config(model_alias=self.config.model_alias)

    @functools.cached_property
    def inference_parameters(self) -> BaseInferenceParameters:
        return self.model_config.inference_parameters

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
