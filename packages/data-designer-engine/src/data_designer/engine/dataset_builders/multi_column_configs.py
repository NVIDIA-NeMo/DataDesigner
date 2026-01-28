# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
from typing import TypeAlias

from pydantic import Field, field_validator

from data_designer.config.base import ConfigBase
from data_designer.config.column_configs import SamplerColumnConfig, SeedDatasetColumnConfig, SingleColumnConfig
from data_designer.config.column_types import ColumnConfigT, DataDesignerColumnType
from data_designer.config.sampler_constraints import ColumnConstraintT
from data_designer.config.seed import SeedConfig


class MultiColumnConfig(ConfigBase, ABC):
    columns: list[SingleColumnConfig] = Field(..., min_length=1)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def column_type(self) -> DataDesignerColumnType:
        return self.columns[0].column_type

    @field_validator("columns", mode="after")
    def validate_column_types_are_the_same(cls, v: list[SingleColumnConfig]) -> list[SingleColumnConfig]:
        if len(set([c.column_type for c in v])) != 1:
            raise ValueError("All column types must be of the same type")
        return v


class SamplerMultiColumnConfig(MultiColumnConfig):
    columns: list[SamplerColumnConfig]
    constraints: list[ColumnConstraintT] = []
    max_rejections_factor: int = 5


class SeedDatasetMultiColumnConfig(SeedConfig, MultiColumnConfig):
    columns: list[SeedDatasetColumnConfig]


DatasetBuilderColumnConfigT: TypeAlias = ColumnConfigT | SeedDatasetMultiColumnConfig | SamplerMultiColumnConfig
