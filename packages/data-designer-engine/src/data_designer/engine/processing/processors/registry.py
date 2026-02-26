# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
from data_designer.plugins.plugin import PluginType
from data_designer.plugins.registry import PluginRegistry


class ProcessorRegistry(TaskRegistry[str, Processor, ConfigBase]): ...


def create_default_processor_registry() -> ProcessorRegistry:
    registry = ProcessorRegistry()
    registry.register(ProcessorType.SCHEMA_TRANSFORM, SchemaTransformProcessor, SchemaTransformProcessorConfig, False)
    registry.register(ProcessorType.DROP_COLUMNS, DropColumnsProcessor, DropColumnsProcessorConfig, False)

    for plugin in PluginRegistry().get_plugins(PluginType.PROCESSOR):
        registry.register(plugin.name, plugin.impl_cls, plugin.config_cls, False)

    return registry
