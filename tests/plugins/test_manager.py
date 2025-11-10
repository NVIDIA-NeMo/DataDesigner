# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from importlib.metadata import EntryPoint
import threading
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest

from data_designer.config.base import ConfigBase
from data_designer.config.column_configs import SingleColumnConfig
from data_designer.engine.configurable_task import ConfigurableTask, ConfigurableTaskMetadata
from data_designer.plugins.errors import PluginNotFoundError, PluginRegistrationError
from data_designer.plugins.manager import PluginManager, _PluginRegistry
from data_designer.plugins.plugin import Plugin, PluginType

# =============================================================================
# Test Stubs
# =============================================================================


class StubPluginConfigA(SingleColumnConfig):
    column_type: Literal["test-plugin-a"] = "test-plugin-a"


class StubPluginConfigB(SingleColumnConfig):
    column_type: Literal["test-plugin-b"] = "test-plugin-b"


class StubPluginTaskA(ConfigurableTask[StubPluginConfigA]):
    @staticmethod
    def metadata() -> ConfigurableTaskMetadata:
        return ConfigurableTaskMetadata(
            name="test_plugin_a",
            description="Test plugin A",
            required_resources=None,
        )


class StubPluginTaskB(ConfigurableTask[StubPluginConfigB]):
    @staticmethod
    def metadata() -> ConfigurableTaskMetadata:
        return ConfigurableTaskMetadata(
            name="test_plugin_b",
            description="Test plugin B",
            required_resources=None,
        )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def plugin_a() -> Plugin:
    return Plugin(
        task_cls=StubPluginTaskA,
        config_cls=StubPluginConfigA,
        plugin_type=PluginType.COLUMN_GENERATOR,
    )


@pytest.fixture
def plugin_b() -> Plugin:
    return Plugin(
        task_cls=StubPluginTaskB,
        config_cls=StubPluginConfigB,
        plugin_type=PluginType.COLUMN_GENERATOR,
    )


@pytest.fixture(autouse=True)
def clean_plugin_manager() -> None:
    """Reset PluginManager singleton state before and after each test."""
    original_instance = PluginManager._instance
    original_discovered = PluginManager._plugins_discovered
    original_plugins = _PluginRegistry._plugins.copy()

    PluginManager._instance = None
    PluginManager._plugins_discovered = False
    _PluginRegistry._plugins = {}

    yield

    PluginManager._instance = original_instance
    PluginManager._plugins_discovered = original_discovered
    _PluginRegistry._plugins = original_plugins


@pytest.fixture
def mock_plugin_discovery():
    """Mock plugin discovery to test with specific entry points."""

    @contextmanager
    def _mock_discovery(entry_points_list):
        with patch("data_designer.plugins.manager.PLUGINS_DISABLED", False):
            with patch("data_designer.plugins.manager.entry_points", return_value=entry_points_list):
                yield

    return _mock_discovery


@pytest.fixture
def mock_entry_points(plugin_a: Plugin, plugin_b: Plugin) -> list[MagicMock]:
    """Create mock entry points for plugin_a and plugin_b."""
    mock_ep_a = MagicMock(spec=EntryPoint)
    mock_ep_a.name = "test-plugin-a"
    mock_ep_a.load.return_value = plugin_a

    mock_ep_b = MagicMock(spec=EntryPoint)
    mock_ep_b.name = "test-plugin-b"
    mock_ep_b.load.return_value = plugin_b

    return [mock_ep_a, mock_ep_b]


# =============================================================================
# _PluginRegistry Tests
# =============================================================================


def test_plugin_registry_register_and_get(plugin_a: Plugin) -> None:
    """Test plugin registration and retrieval."""
    registry = _PluginRegistry()

    registry.register_plugin(plugin_a)

    assert registry.get("test-plugin-a") == plugin_a


def test_plugin_registry_duplicate_raises_error(plugin_a: Plugin) -> None:
    """Test duplicate registration raises PluginRegistrationError."""
    registry = _PluginRegistry()
    registry.register_plugin(plugin_a)

    with pytest.raises(PluginRegistrationError, match="Plugin 'test-plugin-a' already registered"):
        registry.register_plugin(plugin_a)


def test_plugin_registry_get_nonexistent_raises_error() -> None:
    """Test nonexistent plugin raises PluginNotFoundError."""
    registry = _PluginRegistry()

    with pytest.raises(PluginNotFoundError, match="Plugin 'nonexistent' not found"):
        registry.get("nonexistent")


def test_plugin_registry_clear(plugin_a: Plugin, plugin_b: Plugin) -> None:
    """Test clear() removes all plugins."""
    registry = _PluginRegistry()
    registry.register_plugin(plugin_a)
    registry.register_plugin(plugin_b)

    registry.clear()

    with pytest.raises(PluginNotFoundError):
        registry.get("test-plugin-a")
    with pytest.raises(PluginNotFoundError):
        registry.get("test-plugin-b")


# =============================================================================
# PluginManager Singleton Tests
# =============================================================================


def test_plugin_manager_is_singleton(mock_plugin_discovery) -> None:
    """Test PluginManager returns same instance."""
    with mock_plugin_discovery([]):
        manager1 = PluginManager()
        manager2 = PluginManager()

        assert manager1 is manager2


def test_plugin_manager_singleton_thread_safety(mock_plugin_discovery) -> None:
    """Test PluginManager singleton creation is thread-safe."""
    instances: list[PluginManager] = []

    with mock_plugin_discovery([]):

        def create_manager() -> None:
            instances.append(PluginManager())

        threads = [threading.Thread(target=create_manager) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert all(instance is instances[0] for instance in instances)


# =============================================================================
# PluginManager Discovery Tests
# =============================================================================


def test_plugin_manager_discovers_plugins(
    mock_plugin_discovery, mock_entry_points: list[MagicMock], plugin_a: Plugin, plugin_b: Plugin
) -> None:
    """Test PluginManager discovers and loads plugins from entry points."""
    with mock_plugin_discovery(mock_entry_points):
        manager = PluginManager()

        assert manager.num_plugins(PluginType.COLUMN_GENERATOR) == 2
        assert manager.get_plugin("test-plugin-a") == plugin_a
        assert manager.get_plugin("test-plugin-b") == plugin_b


def test_plugin_manager_skips_invalid_plugins(mock_plugin_discovery, plugin_a: Plugin) -> None:
    """Test PluginManager skips non-Plugin objects during discovery."""
    mock_ep_valid = MagicMock(spec=EntryPoint)
    mock_ep_valid.name = "test-plugin-a"
    mock_ep_valid.load.return_value = plugin_a

    mock_ep_invalid = MagicMock(spec=EntryPoint)
    mock_ep_invalid.name = "invalid-plugin"
    mock_ep_invalid.load.return_value = "not a plugin"

    with mock_plugin_discovery([mock_ep_valid, mock_ep_invalid]):
        manager = PluginManager()

        assert manager.num_plugins(PluginType.COLUMN_GENERATOR) == 1
        assert manager.get_plugin("test-plugin-a") == plugin_a


def test_plugin_manager_handles_loading_errors(mock_plugin_discovery, plugin_a: Plugin) -> None:
    """Test PluginManager gracefully handles plugin loading errors."""
    mock_ep_valid = MagicMock(spec=EntryPoint)
    mock_ep_valid.name = "test-plugin-a"
    mock_ep_valid.load.return_value = plugin_a

    mock_ep_error = MagicMock(spec=EntryPoint)
    mock_ep_error.name = "error-plugin"
    mock_ep_error.load.side_effect = Exception("Loading failed")

    with mock_plugin_discovery([mock_ep_valid, mock_ep_error]):
        manager = PluginManager()

        assert manager.num_plugins(PluginType.COLUMN_GENERATOR) == 1
        assert manager.get_plugin("test-plugin-a") == plugin_a


def test_plugin_manager_discovery_runs_once() -> None:
    """Test discovery runs once even with multiple PluginManager instances."""
    mock_entry_points = MagicMock(return_value=[])

    with patch("data_designer.plugins.manager.PLUGINS_DISABLED", False):
        with patch("data_designer.plugins.manager.entry_points", mock_entry_points):
            PluginManager()
            PluginManager()
            PluginManager()

            assert mock_entry_points.call_count == 1


def test_plugin_manager_respects_disabled_flag() -> None:
    """Test PluginManager respects DISABLE_DATA_DESIGNER_PLUGINS flag."""
    mock_entry_points = MagicMock(return_value=[])

    with patch("data_designer.plugins.manager.PLUGINS_DISABLED", True):
        with patch("data_designer.plugins.manager.entry_points", mock_entry_points):
            manager = PluginManager()

            assert mock_entry_points.call_count == 0
            assert manager.num_plugins(PluginType.COLUMN_GENERATOR) == 0


# =============================================================================
# PluginManager Query Methods Tests
# =============================================================================


def test_plugin_manager_get_plugin_raises_error(mock_plugin_discovery) -> None:
    """Test get_plugin() raises error for nonexistent plugin."""
    with mock_plugin_discovery([]):
        manager = PluginManager()

        with pytest.raises(PluginNotFoundError, match="Plugin 'nonexistent' not found"):
            manager.get_plugin("nonexistent")


def test_plugin_manager_get_plugins_by_type(
    mock_plugin_discovery, mock_entry_points: list[MagicMock], plugin_a: Plugin, plugin_b: Plugin
) -> None:
    """Test get_plugins() filters by plugin type."""
    with mock_plugin_discovery(mock_entry_points):
        manager = PluginManager()
        plugins = manager.get_plugins(PluginType.COLUMN_GENERATOR)

        assert len(plugins) == 2
        assert plugin_a in plugins
        assert plugin_b in plugins


def test_plugin_manager_get_plugins_empty(mock_plugin_discovery) -> None:
    """Test get_plugins() returns empty list when no plugins match."""
    with mock_plugin_discovery([]):
        manager = PluginManager()
        plugins = manager.get_plugins(PluginType.COLUMN_GENERATOR)

        assert plugins == []


def test_plugin_manager_get_plugin_names(mock_plugin_discovery, mock_entry_points: list[MagicMock]) -> None:
    """Test get_plugin_names() returns plugin names by type."""
    with mock_plugin_discovery(mock_entry_points):
        manager = PluginManager()
        names = manager.get_plugin_names(PluginType.COLUMN_GENERATOR)

        assert set(names) == {"test-plugin-a", "test-plugin-b"}


# =============================================================================
# PluginManager Type Union Tests
# =============================================================================


def test_plugin_manager_update_type_union(mock_plugin_discovery, mock_entry_points: list[MagicMock]) -> None:
    """Test update_type_union() adds plugin config types to union."""
    with mock_plugin_discovery(mock_entry_points):
        manager = PluginManager()

        type_union: type = ConfigBase
        updated_union = manager.update_type_union(type_union, PluginType.COLUMN_GENERATOR)

        assert StubPluginConfigA in updated_union.__args__
        assert StubPluginConfigB in updated_union.__args__
