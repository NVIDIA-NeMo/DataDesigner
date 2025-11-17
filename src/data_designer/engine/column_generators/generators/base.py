# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import overload

import pandas as pd

from data_designer.config.utils.type_helpers import StrEnum
from data_designer.engine.configurable_task import ConfigurableTask, ConfigurableTaskMetadata, DataT, TaskConfigT


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
