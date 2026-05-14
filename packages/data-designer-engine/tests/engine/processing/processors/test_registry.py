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
from data_designer.plugins.registry import PluginRegistry


def test_create_default_processor_registry() -> None:
    registry = create_default_processor_registry()

    assert isinstance(registry, ProcessorRegistry)
    assert ProcessorType.DROP_COLUMNS in ProcessorRegistry._registry
    assert ProcessorRegistry._registry[ProcessorType.DROP_COLUMNS] == DropColumnsProcessor
    assert ProcessorRegistry._config_registry[ProcessorType.DROP_COLUMNS] == DropColumnsProcessorConfig


def test_processor_plugins_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Cfg(ProcessorConfig):
        processor_type: Literal["test-stub"] = "test-stub"

    class _Impl(Processor[_Cfg]):
        pass

    plugin = MagicMock(impl_cls=_Impl, config_cls=_Cfg)
    plugin.name = "test-stub"
    monkeypatch.setattr(PluginRegistry, "get_plugins", lambda self, pt: [plugin])

    create_default_processor_registry()

    assert ProcessorRegistry._registry["test-stub"] == _Impl
    assert ProcessorRegistry._config_registry["test-stub"] == _Cfg
