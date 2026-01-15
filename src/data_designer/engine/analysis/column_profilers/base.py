# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, model_validator
from typing_extensions import Self

from data_designer.config.base import ConfigBase
from data_designer.config.column_configs import SingleColumnConfig
from data_designer.config.column_types import DataDesignerColumnType
from data_designer.engine.configurable_task import ConfigurableTask, TaskConfigT
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class ColumnConfigWithDataFrame(ConfigBase):
    column_config: SingleColumnConfig
    df: pd.DataFrame

    @model_validator(mode="after")
    def validate_column_exists(self) -> Self:
        if self.column_config.name not in self.df.columns:
            raise ValueError(f"Column {self.column_config.name!r} not found in DataFrame")
        return self

    def as_tuple(self) -> tuple[SingleColumnConfig, pd.DataFrame]:
        return (self.column_config, self.df)


class ColumnProfiler(ConfigurableTask[TaskConfigT], ABC):
    @staticmethod
    @abstractmethod
    def get_applicable_column_types() -> list[DataDesignerColumnType]:
        """Returns a list of column types that this profiler can be applied to during dataset profiling."""

    @abstractmethod
    def profile(self, column_config_with_df: ColumnConfigWithDataFrame) -> BaseModel: ...

    def _initialize(self) -> None:
        logger.info(f"💫 Initializing column profiler: '{self.name}'")
