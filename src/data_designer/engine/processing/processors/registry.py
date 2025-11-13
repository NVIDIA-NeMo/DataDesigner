# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.config.base import ConfigBase
from data_designer.config.processors import (
    DropColumnsProcessorConfig,
    JsonlExportProcessorConfig,
    ProcessorType,
)
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.processing.processors.drop_columns import DropColumnsProcessor
from data_designer.engine.processing.processors.jsonl_export import JsonlExportProcessor
from data_designer.engine.registry.base import TaskRegistry


class ProcessorRegistry(TaskRegistry[str, Processor, ConfigBase]): ...


def create_default_processor_registry() -> ProcessorRegistry:
    registry = ProcessorRegistry()
    registry.register(ProcessorType.DROP_COLUMNS, DropColumnsProcessor, DropColumnsProcessorConfig, False)
    registry.register(ProcessorType.JSONL_EXPORT, JsonlExportProcessor, JsonlExportProcessorConfig, False)
    return registry
