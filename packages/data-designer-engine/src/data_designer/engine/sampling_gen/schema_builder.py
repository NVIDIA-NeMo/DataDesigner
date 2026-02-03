# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

from data_designer.config.column_configs import SamplerColumnConfig
from data_designer.config.sampler_constraints import ColumnConstraintT
from data_designer.config.sampler_params import SamplerParamsT
from data_designer.engine.dataset_builders.multi_column_configs import SamplerMultiColumnConfig
from data_designer.engine.sampling_gen.column import ConditionalDataColumn
from data_designer.engine.sampling_gen.schema import DataSchema


class SchemaBuilder:
    """Builder class for DataSchema objects.

    This class is meant to be a helper for internal usage and experimentation. It
    provides a simple interface for constructing a DataSchema object via `add_column`
    and `add_constraint` methods similar.
    """

    def __init__(
        self,
        columns: list[ConditionalDataColumn] | None = None,
        constraints: list[ColumnConstraintT] | None = None,
    ):
        self._columns = columns or []
        self._constraints = constraints or []

    def add_column(
        self,
        name: str,
        sampler_type: str | None,
        params: dict | SamplerParamsT | None,
        conditional_params: dict[str, SamplerParamsT] | None = None,
        convert_to: str | None = None,
    ) -> None:
        self._columns.append(
            ConditionalDataColumn(
                name=name,
                sampler_type=sampler_type,
                params=params,
                conditional_params=conditional_params or {},
                convert_to=convert_to,
            )
        )

    def add_constraint(self, constraint: ColumnConstraintT) -> None:
        self._constraints.append(constraint)

    def to_sampler_columns(self, max_rejections_factor: int = 5) -> SamplerMultiColumnConfig:
        return SamplerMultiColumnConfig(
            columns=[SamplerColumnConfig(**c.model_dump(mode="json")) for c in self._columns],
            constraints=self._constraints,
            max_rejections_factor=max_rejections_factor,
        )

    def build(self) -> DataSchema:
        return DataSchema(columns=deepcopy(self._columns), constraints=deepcopy(self._constraints))
