# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from contextlib import contextmanager
from enum import Enum
from unittest.mock import patch

from data_designer.engine.testing.stubs import (
    StubPluginConfigModels,
    plugin_blobs_and_seeds,
    plugin_models,
    plugin_models_and_blobs,
)
from data_designer.plugin_manager import PluginManager
from data_designer.plugins.plugin import Plugin
from data_designer.plugins.registry import PluginRegistry


class MockEntryPoint:
    def __init__(self, plugin: Plugin):
        self.plugin = plugin

    def load(self) -> Plugin:
        return self.plugin


@contextmanager
def mock_plugin_system(plugins: list[Plugin]) -> Generator[None, None, None]:
    """Context manager to mock plugin entry points to return the provided plugins.

    This works regardless of whether the actual environment has plugins available or not
    by patching at the module level where PluginManager is instantiated.
    """
    mock_entry_points = [MockEntryPoint(plugin) for plugin in plugins]
    with (
        patch("data_designer.plugins.registry.entry_points", return_value=mock_entry_points),
        patch("data_designer.plugins.registry.PLUGINS_DISABLED", False),
    ):
        yield
    PluginRegistry.reset()


def make_test_enum(plugins: list[Plugin]) -> type[Enum]:
    TestEnum = Enum("TestEnum", {plugin.name.replace("-", "_").upper(): plugin.name for plugin in plugins}, type=str)
    return TestEnum


def test_get_column_generator_plugins_with_plugins() -> None:
    """Test getting plugin column configs when plugins are available."""
    with mock_plugin_system([plugin_blobs_and_seeds, plugin_models]):
        manager = PluginManager()
        result = manager.get_column_generator_plugins()

    assert len(result) == 2
    assert [p.name for p in result] == [plugin_blobs_and_seeds.name, plugin_models.name]


def test_get_column_generator_plugins_empty() -> None:
    """Test getting plugin column configs when no plugins are registered."""
    with mock_plugin_system([]):
        manager = PluginManager()
        result = manager.get_column_generator_plugins()

    assert result == []


def test_get_column_generator_plugin_if_exists_found() -> None:
    """Test getting a specific plugin by name when it exists."""
    with mock_plugin_system([plugin_models]):
        manager = PluginManager()
        result = manager.get_column_generator_plugin_if_exists(plugin_models.name)

    assert result == plugin_models


def test_get_column_generator_plugin_if_exists_not_found() -> None:
    """Test getting a specific plugin by name when it doesn't exist."""
    with mock_plugin_system([]):
        manager = PluginManager()
        result = manager.get_column_generator_plugin_if_exists(plugin_models.name)

    assert result is None


def test_get_plugin_column_types_with_plugins() -> None:
    """Test getting plugin column types when plugins are available."""
    all_plugins = [plugin_models, plugin_models_and_blobs, plugin_blobs_and_seeds]
    TestEnum = make_test_enum(all_plugins)
    with mock_plugin_system(all_plugins):
        manager = PluginManager()
        result = manager.get_plugin_column_types(TestEnum)

    assert len(result) == 3
    assert all(isinstance(item, TestEnum) for item in result)


def test_get_plugin_column_types_empty() -> None:
    """Test getting plugin column types when no plugins are registered."""
    TestEnum = make_test_enum([])

    with mock_plugin_system([]):
        manager = PluginManager()
        result = manager.get_plugin_column_types(TestEnum)

    assert result == []


def test_inject_into_column_config_type_union_with_plugins() -> None:
    """Test injecting plugins into column config type union."""

    class BaseType1:
        pass

    class BaseType2:
        pass

    TestUnion = BaseType1 | BaseType2

    with mock_plugin_system([plugin_models]):
        manager = PluginManager()
        result = manager.inject_into_column_config_type_union(TestUnion)

    assert result == BaseType1 | BaseType2 | StubPluginConfigModels
