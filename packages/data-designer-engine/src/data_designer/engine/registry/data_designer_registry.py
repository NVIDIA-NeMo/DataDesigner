# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.engine.analysis.column_profilers.registry import (
    ColumnProfilerRegistry,
    create_default_column_profiler_registry,
)
from data_designer.engine.column_generators.registry import (
    ColumnGeneratorRegistry,
    create_default_column_generator_registry,
)
from data_designer.engine.processing.processors.registry import ProcessorRegistry, create_default_processor_registry


class DataDesignerRegistry:
    def __init__(
        self,
        *,
        column_generator_registry: ColumnGeneratorRegistry | None = None,
        column_profiler_registry: ColumnProfilerRegistry | None = None,
        processor_registry: ProcessorRegistry | None = None,
    ):
        self._column_generator_registry = column_generator_registry or create_default_column_generator_registry()
        self._column_profiler_registry = column_profiler_registry or create_default_column_profiler_registry()
        self._processor_registry = processor_registry or create_default_processor_registry()

    @property
    def column_generators(self) -> ColumnGeneratorRegistry:
        return self._column_generator_registry

    @property
    def column_profilers(self) -> ColumnProfilerRegistry:
        return self._column_profiler_registry

    @property
    def processors(self) -> ProcessorRegistry:
        return self._processor_registry
