# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal
from unittest.mock import MagicMock

import pytest

from data_designer.config.processors import DropColumnsProcessorConfig, ProcessorConfig, ProcessorType
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.processing.processors.drop_columns import DropColumnsProcessor
from data_designer.engine.processing.processors.registry import (
    ProcessorRegistry,
    create_default_processor_registry,
)
from data_designer.plugins.plugin import PluginType
from data_designer.plugins.registry import PluginRegistry


def test_create_default_processor_registry() -> None:
    registry = create_default_processor_registry()

    assert isinstance(registry, ProcessorRegistry)
    assert ProcessorType.DROP_COLUMNS in ProcessorRegistry._registry
    assert ProcessorRegistry._registry[ProcessorType.DROP_COLUMNS] == DropColumnsProcessor
    assert ProcessorRegistry._config_registry[ProcessorType.DROP_COLUMNS] == DropColumnsProcessorConfig


def test_processor_plugins_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubConfig(ProcessorConfig):
        processor_type: Literal["test-stub"] = "test-stub"

    class _StubProcessor(Processor[_StubConfig]):
        pass

    plugin = MagicMock()
    plugin.name = "test-stub"
    plugin.impl_cls = _StubProcessor
    plugin.config_cls = _StubConfig

    original_get_plugins = PluginRegistry.get_plugins
    monkeypatch.setattr(
        PluginRegistry,
        "get_plugins",
        lambda self, pt: [plugin] if pt == PluginType.PROCESSOR else original_get_plugins(self, pt),
    )

    try:
        create_default_processor_registry()

        assert ProcessorRegistry._registry["test-stub"] == _StubProcessor
        assert ProcessorRegistry._config_registry["test-stub"] == _StubConfig
    finally:
        ProcessorRegistry._registry.pop("test-stub", None)
        ProcessorRegistry._reverse_registry.pop(_StubProcessor, None)
        ProcessorRegistry._config_registry.pop("test-stub", None)
        ProcessorRegistry._reverse_config_registry.pop(_StubConfig, None)
