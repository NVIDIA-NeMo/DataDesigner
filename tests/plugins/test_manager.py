# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Literal

import pytest

from data_designer.config.base import ConfigBase
from data_designer.config.column_configs import SingleColumnConfig
from data_designer.engine.configurable_task import ConfigurableTask, ConfigurableTaskMetadata
from data_designer.plugins.errors import PluginNotFoundError, PluginRegistrationError
from data_designer.plugins.manager import PluginManager, _PluginRegistry
from data_designer.plugins.plugin import Plugin, PluginType

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def plugin_manager() -> PluginManager:
    """Create a PluginManager with a clean registry.

    This fixture ensures the singleton registry is cleared before and after each test,
    preventing state leakage between tests and from any plugins in the default
    plugin directory.
    """
    manager = PluginManager()
    # Clear any plugins that may have been auto-discovered (e.g., from ~/.data_designer/plugins/)
    manager.registry.clear()
    yield manager
    # Cleanup: clear the singleton registry after the test
    manager.registry.clear()


def create_plugin_file(
    dir_path: Path,
    filename: str,
    plugin_name: str,
    column_type: str,
    task_name: str | None = None,
) -> Path:
    """Helper to create test plugin files with less boilerplate.

    Args:
        dir_path: Directory to create the plugin file in
        filename: Name of the plugin file (e.g., "test_plugin.py")
        plugin_name: Name of the plugin (e.g., "MyPlugin")
        column_type: Column type literal value (e.g., "my-plugin")
        task_name: Task metadata name (defaults to plugin_name lowercase with underscores)

    Returns:
        Path to the created plugin file
    """
    if task_name is None:
        task_name = plugin_name.lower().replace("-", "_")

    plugin_var_name = plugin_name.lower().replace("-", "_")

    plugin_file = dir_path / filename
    plugin_file.write_text(
        f"""
from typing import Literal
from data_designer.config.column_configs import SingleColumnConfig
from data_designer.engine.configurable_task import ConfigurableTask, ConfigurableTaskMetadata
from data_designer.plugins.plugin import Plugin, PluginType


class {plugin_name}Config(SingleColumnConfig):
    column_type: Literal["{column_type}"] = "{column_type}"
    name: str


class {plugin_name}Task(ConfigurableTask[{plugin_name}Config]):
    @staticmethod
    def metadata() -> ConfigurableTaskMetadata:
        return ConfigurableTaskMetadata(
            name="{task_name}",
            description="{plugin_name} task",
            required_resources=None,
        )


{plugin_var_name} = Plugin(
    task_cls={plugin_name}Task,
    config_cls={plugin_name}Config,
    plugin_type=PluginType.COLUMN_GENERATOR,
)
"""
    )
    return plugin_file


@pytest.fixture
def temp_plugin_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with a test plugin file."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    create_plugin_file(plugin_dir, "test_plugin.py", "MyPlugin", "my-plugin")
    return plugin_dir


@pytest.fixture
def invalid_plugin_dir(tmp_path: Path) -> Path:
    """Create a directory with an invalid plugin file."""
    plugin_dir = tmp_path / "invalid_plugins"
    plugin_dir.mkdir()

    invalid_file = plugin_dir / "invalid.py"
    invalid_file.write_text("import syntax error here")

    return plugin_dir


# =============================================================================
# Plugin Discovery Tests
# =============================================================================


def test_discover_finds_plugin(plugin_manager: PluginManager, temp_plugin_dir: Path) -> None:
    """Test that discover() finds and registers plugins in the plugin directory."""
    plugin_manager.discover(plugin_dir=temp_plugin_dir)

    assert plugin_manager.num_plugins(PluginType.COLUMN_GENERATOR) == 1
    assert "my-plugin" in plugin_manager.get_plugin_names(PluginType.COLUMN_GENERATOR)


@pytest.mark.parametrize(
    "dir_setup",
    [
        ("empty", lambda tmp_path: (tmp_path / "empty").mkdir() or (tmp_path / "empty")),
        ("nonexistent", lambda tmp_path: tmp_path / "does_not_exist"),
    ],
    ids=["empty_directory", "nonexistent_directory"],
)
def test_discover_handles_missing_or_empty_directories(
    plugin_manager: PluginManager, tmp_path: Path, dir_setup: tuple[str, callable]
) -> None:
    """Test that discover() handles empty and nonexistent directories gracefully."""
    _, setup_func = dir_setup
    plugin_dir = setup_func(tmp_path)

    plugin_manager.discover(plugin_dir=plugin_dir)

    assert plugin_manager.num_plugins(PluginType.COLUMN_GENERATOR) == 0


def test_discover_skips_private_files(plugin_manager: PluginManager, tmp_path: Path) -> None:
    """Test that discover() skips files starting with underscore."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    create_plugin_file(plugin_dir, "_private_plugin.py", "PrivatePlugin", "private-plugin")

    plugin_manager.discover(plugin_dir=plugin_dir)

    assert plugin_manager.num_plugins(PluginType.COLUMN_GENERATOR) == 0


def test_discover_handles_invalid_files(plugin_manager: PluginManager, invalid_plugin_dir: Path) -> None:
    """Test that discover() gracefully handles invalid Python files."""
    plugin_manager.discover(plugin_dir=invalid_plugin_dir)

    assert plugin_manager.num_plugins(PluginType.COLUMN_GENERATOR) == 0


def test_discover_finds_multiple_plugins_in_same_file(plugin_manager: PluginManager, tmp_path: Path) -> None:
    """Test that discover() can find multiple Plugin instances in the same file."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    multi_plugin_file = plugin_dir / "multi.py"
    multi_plugin_file.write_text(
        """
from typing import Literal
from data_designer.config.column_configs import SingleColumnConfig
from data_designer.engine.configurable_task import ConfigurableTask, ConfigurableTaskMetadata
from data_designer.plugins.plugin import Plugin, PluginType


class Plugin1Config(SingleColumnConfig):
    column_type: Literal["plugin-1"] = "plugin-1"
    name: str


class Plugin1Task(ConfigurableTask[Plugin1Config]):
    @staticmethod
    def metadata() -> ConfigurableTaskMetadata:
        return ConfigurableTaskMetadata(
            name="plugin_1",
            description="Plugin 1",
            required_resources=None,
        )


class Plugin2Config(SingleColumnConfig):
    column_type: Literal["plugin-2"] = "plugin-2"
    name: str


class Plugin2Task(ConfigurableTask[Plugin2Config]):
    @staticmethod
    def metadata() -> ConfigurableTaskMetadata:
        return ConfigurableTaskMetadata(
            name="plugin_2",
            description="Plugin 2",
            required_resources=None,
        )


plugin1 = Plugin(
    task_cls=Plugin1Task,
    config_cls=Plugin1Config,
    plugin_type=PluginType.COLUMN_GENERATOR,
)

plugin2 = Plugin(
    task_cls=Plugin2Task,
    config_cls=Plugin2Config,
    plugin_type=PluginType.COLUMN_GENERATOR,
)
"""
    )

    plugin_manager.discover(plugin_dir=plugin_dir)

    assert plugin_manager.num_plugins(PluginType.COLUMN_GENERATOR) == 2
    plugin_names = plugin_manager.get_plugin_names(PluginType.COLUMN_GENERATOR)
    assert "plugin-1" in plugin_names
    assert "plugin-2" in plugin_names


def test_discover_recursive_search(plugin_manager: PluginManager, tmp_path: Path) -> None:
    """Test that discover() recursively searches subdirectories."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    subdir = plugin_dir / "subdir"
    subdir.mkdir()

    create_plugin_file(subdir, "nested.py", "NestedPlugin", "nested-plugin")

    plugin_manager.discover(plugin_dir=plugin_dir)

    assert plugin_manager.num_plugins(PluginType.COLUMN_GENERATOR) == 1
    assert "nested-plugin" in plugin_manager.get_plugin_names(PluginType.COLUMN_GENERATOR)


def test_discover_multiple_calls(plugin_manager: PluginManager, tmp_path: Path) -> None:
    """Test that discover() can be called multiple times to discover plugins from different directories."""
    dir1 = tmp_path / "plugins1"
    dir1.mkdir()
    create_plugin_file(dir1, "plugin1.py", "Plugin1", "plugin-1")

    dir2 = tmp_path / "plugins2"
    dir2.mkdir()
    create_plugin_file(dir2, "plugin2.py", "Plugin2", "plugin-2")

    plugin_manager.discover(plugin_dir=dir1)
    plugin_manager.discover(plugin_dir=dir2)

    assert plugin_manager.num_plugins(PluginType.COLUMN_GENERATOR) == 2
    plugin_names = plugin_manager.get_plugin_names(PluginType.COLUMN_GENERATOR)
    assert "plugin-1" in plugin_names
    assert "plugin-2" in plugin_names


# =============================================================================
# Plugin Retrieval Tests
# =============================================================================


def test_get_plugin_returns_correct_plugin(plugin_manager: PluginManager, temp_plugin_dir: Path) -> None:
    """Test that get_plugin() returns the correct plugin by name."""
    plugin_manager.discover(plugin_dir=temp_plugin_dir)

    plugin = plugin_manager.get_plugin("my-plugin")

    assert plugin.name == "my-plugin"
    assert plugin.plugin_type == PluginType.COLUMN_GENERATOR
    assert plugin.config_cls.__name__ == "MyPluginConfig"
    assert plugin.task_cls.__name__ == "MyPluginTask"


def test_get_plugin_raises_not_found_error(plugin_manager: PluginManager) -> None:
    """Test that get_plugin() raises PluginNotFoundError for nonexistent plugins."""
    with pytest.raises(PluginNotFoundError, match="Plugin 'nonexistent' not found"):
        plugin_manager.get_plugin("nonexistent")


def test_get_plugins_returns_plugins_by_type(plugin_manager: PluginManager, tmp_path: Path) -> None:
    """Test that get_plugins() returns all plugins of a specific type."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    create_plugin_file(plugin_dir, "plugin1.py", "Plugin1", "plugin-1")
    create_plugin_file(plugin_dir, "plugin2.py", "Plugin2", "plugin-2")

    plugin_manager.discover(plugin_dir=plugin_dir)

    plugins = plugin_manager.get_plugins(PluginType.COLUMN_GENERATOR)

    assert len(plugins) == 2
    plugin_names = [p.name for p in plugins]
    assert "plugin-1" in plugin_names
    assert "plugin-2" in plugin_names


def test_get_plugin_names_returns_all_names(plugin_manager: PluginManager, temp_plugin_dir: Path) -> None:
    """Test that get_plugin_names() returns all plugin names for a given type."""
    plugin_manager.discover(plugin_dir=temp_plugin_dir)

    names = plugin_manager.get_plugin_names(PluginType.COLUMN_GENERATOR)

    assert names == ["my-plugin"]


def test_num_plugins_returns_count(plugin_manager: PluginManager, temp_plugin_dir: Path) -> None:
    """Test that num_plugins() returns the correct count."""
    assert plugin_manager.num_plugins(PluginType.COLUMN_GENERATOR) == 0

    plugin_manager.discover(plugin_dir=temp_plugin_dir)
    assert plugin_manager.num_plugins(PluginType.COLUMN_GENERATOR) == 1


# =============================================================================
# Type Union Tests
# =============================================================================


def test_update_type_union_adds_config_types(plugin_manager: PluginManager, temp_plugin_dir: Path) -> None:
    """Test that update_type_union() adds plugin config classes to the type union."""
    plugin_manager.discover(plugin_dir=temp_plugin_dir)

    # Start with a basic type
    type_union = SingleColumnConfig

    updated_union = plugin_manager.update_type_union(type_union, PluginType.COLUMN_GENERATOR)

    # The union should now include the plugin's config class
    plugin = plugin_manager.get_plugin("my-plugin")
    assert plugin.config_cls in updated_union.__args__


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_register_duplicate_plugin_raises_error(plugin_manager: PluginManager, temp_plugin_dir: Path) -> None:
    """Test that registering a duplicate plugin raises PluginRegistrationError."""
    plugin_manager.discover(plugin_dir=temp_plugin_dir)

    # Try to discover the same plugin again
    with pytest.raises(PluginRegistrationError, match="Plugin 'my-plugin' already registered"):
        plugin_manager.discover(plugin_dir=temp_plugin_dir)


# =============================================================================
# Plugin Validation Tests
# =============================================================================


def test_plugin_with_invalid_discriminator_field() -> None:
    """Test that Plugin validation fails when discriminator field is missing."""

    class InvalidConfig(ConfigBase):
        name: str

    class InvalidTask(ConfigurableTask[InvalidConfig]):
        @staticmethod
        def metadata() -> ConfigurableTaskMetadata:
            return ConfigurableTaskMetadata(
                name="invalid",
                description="Invalid plugin",
                required_resources=None,
            )

    with pytest.raises(ValueError, match="Discriminator field 'column_type' not found"):
        Plugin(
            task_cls=InvalidTask,
            config_cls=InvalidConfig,
            plugin_type=PluginType.COLUMN_GENERATOR,
        )


def test_plugin_with_non_literal_discriminator() -> None:
    """Test that Plugin validation fails when discriminator field is not a Literal type."""

    class NonLiteralConfig(SingleColumnConfig):
        column_type: str = "non-literal"  # Should be Literal["non-literal"]
        name: str

    class NonLiteralTask(ConfigurableTask[NonLiteralConfig]):
        @staticmethod
        def metadata() -> ConfigurableTaskMetadata:
            return ConfigurableTaskMetadata(
                name="non_literal",
                description="Non-literal plugin",
                required_resources=None,
            )

    with pytest.raises(ValueError, match="Field 'column_type' .* must be a Literal type"):
        Plugin(
            task_cls=NonLiteralTask,
            config_cls=NonLiteralConfig,
            plugin_type=PluginType.COLUMN_GENERATOR,
        )


def test_plugin_with_non_string_discriminator_default() -> None:
    """Test that Plugin validation fails when discriminator default is not a string."""

    class NonStringConfig(ConfigBase):
        column_type: Literal[123] = 123  # Should be a string
        name: str

    class NonStringTask(ConfigurableTask[NonStringConfig]):
        @staticmethod
        def metadata() -> ConfigurableTaskMetadata:
            return ConfigurableTaskMetadata(
                name="non_string",
                description="Non-string plugin",
                required_resources=None,
            )

    with pytest.raises(ValueError, match="The default of 'column_type' must be a string"):
        Plugin(
            task_cls=NonStringTask,
            config_cls=NonStringConfig,
            plugin_type=PluginType.COLUMN_GENERATOR,
        )


def test_plugin_with_invalid_enum_key() -> None:
    """Test that Plugin validation fails when discriminator can't be converted to valid enum key."""

    class InvalidEnumKeyConfig(SingleColumnConfig):
        column_type: Literal["123-invalid"] = "123-invalid"  # Starts with number
        name: str

    class InvalidEnumKeyTask(ConfigurableTask[InvalidEnumKeyConfig]):
        @staticmethod
        def metadata() -> ConfigurableTaskMetadata:
            return ConfigurableTaskMetadata(
                name="invalid_enum",
                description="Invalid enum key plugin",
                required_resources=None,
            )

    with pytest.raises(ValueError, match="cannot be converted to a valid enum key"):
        Plugin(
            task_cls=InvalidEnumKeyTask,
            config_cls=InvalidEnumKeyConfig,
            plugin_type=PluginType.COLUMN_GENERATOR,
        )


def test_plugin_name_property(plugin_manager: PluginManager, temp_plugin_dir: Path) -> None:
    """Test that plugin name property correctly extracts name from discriminator field."""
    plugin_manager.discover(plugin_dir=temp_plugin_dir)

    plugin = plugin_manager.get_plugin("my-plugin")
    assert plugin.name == "my-plugin"


def test_plugin_enum_key_property(plugin_manager: PluginManager, temp_plugin_dir: Path) -> None:
    """Test that plugin enum_key property correctly converts name to enum format."""
    plugin_manager.discover(plugin_dir=temp_plugin_dir)

    plugin = plugin_manager.get_plugin("my-plugin")
    assert plugin.enum_key == "MY_PLUGIN"


# =============================================================================
# Registry Singleton Tests
# =============================================================================


def test_registry_is_singleton(plugin_manager: PluginManager) -> None:
    """Test that _PluginRegistry is a singleton."""
    registry1 = _PluginRegistry()
    registry2 = _PluginRegistry()

    assert registry1 is registry2
    assert registry1 is plugin_manager.registry


def test_registry_clear_affects_all_instances(plugin_manager: PluginManager, temp_plugin_dir: Path) -> None:
    """Test that clearing registry affects all manager instances."""
    plugin_manager.discover(plugin_dir=temp_plugin_dir)

    assert plugin_manager.num_plugins(PluginType.COLUMN_GENERATOR) == 1

    manager2 = PluginManager()
    assert manager2.num_plugins(PluginType.COLUMN_GENERATOR) == 1

    plugin_manager.registry.clear()

    assert plugin_manager.num_plugins(PluginType.COLUMN_GENERATOR) == 0
    assert manager2.num_plugins(PluginType.COLUMN_GENERATOR) == 0


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_plugin_workflow(plugin_manager: PluginManager, tmp_path: Path) -> None:
    """Test complete workflow: discover → retrieve → validate plugin properties."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    create_plugin_file(plugin_dir, "workflow_plugin.py", "WorkflowPlugin", "workflow-plugin")

    plugin_manager.discover(plugin_dir=plugin_dir)

    assert plugin_manager.num_plugins(PluginType.COLUMN_GENERATOR) == 1
    assert "workflow-plugin" in plugin_manager.get_plugin_names(PluginType.COLUMN_GENERATOR)

    plugin = plugin_manager.get_plugin("workflow-plugin")
    assert plugin.name == "workflow-plugin"
    assert plugin.enum_key == "WORKFLOW_PLUGIN"
    assert plugin.plugin_type == PluginType.COLUMN_GENERATOR
    assert plugin.config_cls.__name__ == "WorkflowPluginConfig"
    assert plugin.task_cls.__name__ == "WorkflowPluginTask"
