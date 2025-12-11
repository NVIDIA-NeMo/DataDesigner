# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.config.base import ConfigBase
from data_designer.config.processors import (
    DropColumnsProcessorConfig,
    ProcessorType,
    SchemaTransformProcessorConfig,
)
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.processing.processors.drop_columns import DropColumnsProcessor
from data_designer.engine.processing.processors.schema_transform import SchemaTransformProcessor
from data_designer.engine.registry.base import TaskRegistry


class ProcessorRegistry(TaskRegistry[str, Processor, ConfigBase]): ...


def create_default_processor_registry() -> ProcessorRegistry:
    registry = ProcessorRegistry()
    registry.register(ProcessorType.SCHEMA_TRANSFORM, SchemaTransformProcessor, SchemaTransformProcessorConfig, False)
    registry.register(ProcessorType.DROP_COLUMNS, DropColumnsProcessor, DropColumnsProcessorConfig, False)
    return registry
