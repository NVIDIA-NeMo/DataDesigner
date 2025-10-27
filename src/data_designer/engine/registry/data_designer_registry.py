# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.engine.analysis.column_profilers.registry import (
    ColumnProfilerRegistry,
    create_default_column_profiler_registry,
)
from data_designer.engine.column_generators.registry import (
    ColumnGeneratorRegistry,
    create_default_column_generator_registry,
)


class DataDesignerRegistry:
    def __init__(
        self,
        *,
        column_generator_registry: ColumnGeneratorRegistry | None = None,
        column_profiler_registry: ColumnProfilerRegistry | None = None,
    ):
        self._column_generator_registry = column_generator_registry or create_default_column_generator_registry()
        self._column_profiler_registry = column_profiler_registry or create_default_column_profiler_registry()

    @property
    def column_generators(self) -> ColumnGeneratorRegistry:
        return self._column_generator_registry

    @property
    def column_profilers(self) -> ColumnProfilerRegistry:
        return self._column_profiler_registry
