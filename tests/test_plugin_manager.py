# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from contextlib import contextmanager
from enum import Enum
from unittest.mock import MagicMock, Mock, patch

import pytest

from data_designer.plugin_manager import PluginManager


class MockPluginType(str, Enum):
    """Mock PluginType enum for testing."""

    COLUMN_GENERATOR = "column-generator"

    @property
    def discriminator_field(self) -> str:
        return "column_type"


def create_mock_plugin(name: str, plugin_type: MockPluginType, resources: list[str] | None = None) -> Mock:
    """Create a mock plugin with specified name and resources.

    Args:
        name: Plugin name.
        plugin_type: Plugin type enum.
        resources: List of required resources, or None if no resource requirements.

    Returns:
        Mock plugin object.
    """
    plugin = Mock()
    plugin.name = name
    plugin.plugin_type = plugin_type
    plugin.config_cls = Mock(name=name)

    mock_task = Mock()
    mock_task.metadata = Mock(return_value=Mock(required_resources=resources))
    plugin.task_cls = mock_task

    return plugin


@contextmanager
def mock_plugin_system(registry: MagicMock) -> Generator[None, None, None]:
    """Context manager to mock the plugin system with a given registry.

    This works regardless of whether the actual environment has plugins available or not
    by patching at the module level where PluginManager is instantiated.
    """
    with patch("data_designer.plugin_manager.PluginRegistry", return_value=registry, create=True):
        with patch("data_designer.plugin_manager.PluginType", MockPluginType, create=True):
            yield


@pytest.fixture
def mock_plugin_registry() -> MagicMock:
    """Create a mock plugin registry."""
    return MagicMock()


@pytest.fixture
def mock_plugins() -> list[Mock]:
    """Create mock plugins for testing."""
    return [
        create_mock_plugin("plugin-one", MockPluginType.COLUMN_GENERATOR, ["resource1", "resource2"]),
        create_mock_plugin("plugin-two", MockPluginType.COLUMN_GENERATOR, ["resource1"]),
        create_mock_plugin("plugin-three", MockPluginType.COLUMN_GENERATOR, ["resource2", "resource3"]),
    ]


def test_get_column_generator_plugins_with_plugins(mock_plugin_registry: MagicMock, mock_plugins: list[Mock]) -> None:
    """Test getting plugin column configs when plugins are available."""
    mock_plugin_registry.get_plugins.return_value = [mock_plugins[0], mock_plugins[1]]

    with mock_plugin_system(mock_plugin_registry):
        manager = PluginManager()
        result = manager.get_column_generator_plugins()

    assert len(result) == 2
    assert [p.name for p in result] == ["plugin-one", "plugin-two"]
    mock_plugin_registry.get_plugins.assert_called_once_with(MockPluginType.COLUMN_GENERATOR)


def test_get_column_generator_plugins_empty(mock_plugin_registry: MagicMock) -> None:
    """Test getting plugin column configs when no plugins are registered."""
    mock_plugin_registry.get_plugins.return_value = []
    with mock_plugin_system(mock_plugin_registry):
        manager = PluginManager()
        result = manager.get_column_generator_plugins()

    assert result == []


def test_get_column_generator_plugin_if_exists_found(mock_plugin_registry: MagicMock, mock_plugins: list[Mock]) -> None:
    """Test getting a specific plugin by name when it exists."""
    mock_plugin_registry.plugin_exists.return_value = True
    mock_plugin_registry.get_plugin.return_value = mock_plugins[0]

    with mock_plugin_system(mock_plugin_registry):
        manager = PluginManager()
        result = manager.get_column_generator_plugin_if_exists("plugin-one")

    assert result is not None
    assert result.name == "plugin-one"
    mock_plugin_registry.plugin_exists.assert_called_once_with("plugin-one")
    mock_plugin_registry.get_plugin.assert_called_once_with("plugin-one")


def test_get_column_generator_plugin_if_exists_not_found(mock_plugin_registry: MagicMock) -> None:
    """Test getting a specific plugin by name when it doesn't exist."""
    mock_plugin_registry.plugin_exists.return_value = False

    with mock_plugin_system(mock_plugin_registry):
        manager = PluginManager()
        result = manager.get_column_generator_plugin_if_exists("plugin-three")

    assert result is None
    mock_plugin_registry.plugin_exists.assert_called_once_with("plugin-three")
    mock_plugin_registry.get_plugin.assert_not_called()


def test_get_plugin_column_types_with_plugins(mock_plugin_registry: MagicMock, mock_plugins: list[Mock]) -> None:
    """Test getting plugin column types when plugins are available."""
    TestEnum = Enum(
        "TestEnum", {plugin.name.replace("-", "_").upper(): plugin.name for plugin in mock_plugins}, type=str
    )
    mock_plugin_registry.get_plugins.return_value = mock_plugins

    with mock_plugin_system(mock_plugin_registry):
        manager = PluginManager()
        result = manager.get_plugin_column_types(TestEnum)

    assert len(result) == 3
    assert all(isinstance(item, TestEnum) for item in result)
    mock_plugin_registry.get_plugins.assert_called_once_with(MockPluginType.COLUMN_GENERATOR)


def test_get_plugin_column_types_with_resource_filtering(
    mock_plugin_registry: MagicMock, mock_plugins: list[Mock]
) -> None:
    """Test filtering plugins by required resources."""
    TestEnum = Enum(
        "TestEnum", {"PLUGIN_ONE": "plugin-one", "PLUGIN_TWO": "plugin-two", "PLUGIN_THREE": "plugin-three"}, type=str
    )
    mock_plugin_registry.get_plugins.return_value = mock_plugins

    with mock_plugin_system(mock_plugin_registry):
        manager = PluginManager()
        result = manager.get_plugin_column_types(TestEnum, required_resources=["resource1"])

    assert len(result) == 2
    assert set(result) == {TestEnum.PLUGIN_ONE, TestEnum.PLUGIN_TWO}


def test_get_plugin_column_types_filters_none_resources(mock_plugin_registry: MagicMock) -> None:
    """Test filtering when plugin has None for required_resources."""
    plugin = create_mock_plugin("plugin-one", MockPluginType.COLUMN_GENERATOR, None)
    TestEnum = Enum("TestEnum", {"PLUGIN_ONE": "plugin-one"}, type=str)
    mock_plugin_registry.get_plugins.return_value = [plugin]

    with mock_plugin_system(mock_plugin_registry):
        manager = PluginManager()
        result = manager.get_plugin_column_types(TestEnum, required_resources=["resource1"])

    assert result == []


def test_get_plugin_column_types_empty(mock_plugin_registry: MagicMock) -> None:
    """Test getting plugin column types when no plugins are registered."""
    TestEnum = Enum("TestEnum", {}, type=str)

    mock_plugin_registry.get_plugins.return_value = []
    with mock_plugin_system(mock_plugin_registry):
        manager = PluginManager()
        result = manager.get_plugin_column_types(TestEnum)

    assert result == []


def test_inject_into_column_config_type_union_with_plugins(mock_plugin_registry: MagicMock) -> None:
    """Test injecting plugins into column config type union."""

    class BaseType:
        pass

    mock_plugin_registry.add_plugin_types_to_union.return_value = str | int

    with mock_plugin_system(mock_plugin_registry):
        manager = PluginManager()
        result = manager.inject_into_column_config_type_union(BaseType)

    assert result == str | int
    mock_plugin_registry.add_plugin_types_to_union.assert_called_once_with(BaseType, MockPluginType.COLUMN_GENERATOR)
