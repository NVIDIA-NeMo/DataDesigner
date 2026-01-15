# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.config.processors import DropColumnsProcessorConfig, ProcessorType
from data_designer.engine.processing.processors.drop_columns import DropColumnsProcessor
from data_designer.engine.processing.processors.registry import (
    ProcessorRegistry,
    create_default_processor_registry,
)


def test_create_default_processor_registry():
    registry = create_default_processor_registry()

    assert isinstance(registry, ProcessorRegistry)
    assert ProcessorType.DROP_COLUMNS in ProcessorRegistry._registry
    assert ProcessorRegistry._registry[ProcessorType.DROP_COLUMNS] == DropColumnsProcessor
    assert ProcessorRegistry._config_registry[ProcessorType.DROP_COLUMNS] == DropColumnsProcessorConfig
