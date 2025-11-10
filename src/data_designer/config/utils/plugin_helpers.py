from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Type, TypeAlias

from .misc import can_run_data_designer_locally

if TYPE_CHECKING:
    from data_designer.plugins.manager import PluginManager
    from data_designer.plugins.plugin import Plugin


plugin_manager = None
if can_run_data_designer_locally():
    from data_designer.plugins.manager import PluginManager, PluginType

    plugin_manager = PluginManager()


def get_plugin_column_configs() -> list[Plugin]:
    """Get all plugin column configs.

    Returns:
        A list of all plugin column configs.
    """
    if plugin_manager:
        return [
            plugin_manager.get_plugin(plugin_name)
            for plugin_name in plugin_manager.get_plugin_names(PluginType.COLUMN_GENERATOR)
        ]
    return []


def get_plugin_column_config_if_available(plugin_name: str) -> Plugin | None:
    """Get a plugin column config by name if available.

    Args:
        plugin_name: The name of the plugin to retrieve.

    Returns:
        The plugin if found, otherwise None.
    """
    if plugin_manager:
        for name in plugin_manager.get_plugin_names(PluginType.COLUMN_GENERATOR):
            if plugin_name == name:
                return plugin_manager.get_plugin(plugin_name)
    return None


def get_plugin_column_types(enum_type: Type[Enum], required_resources: list[str] | None = None) -> list[Enum]:
    """Get a list of plugin column types.

    Args:
        enum_type: The enum type to use for plugin entries.
        required_resources: If provided, only return plugins with the required resources.

    Returns:
        A list of plugin column types.
    """
    type_list = []
    if plugin_manager:
        for plugin in plugin_manager.get_plugins(PluginType.COLUMN_GENERATOR):
            if required_resources:
                task_required_resources = plugin.task_cls.metadata().required_resources or []
                if not all(resource in task_required_resources for resource in required_resources):
                    continue
            type_list.append(enum_type(plugin.name))
    return type_list


def inject_into_column_config_type_union(column_config_type: Type[TypeAlias]) -> Type[TypeAlias]:
    """Inject plugins into the column config type.

    Args:
        column_config_type: The column config type to inject plugins into.

    Returns:
        The column config type with plugins injected.
    """
    if plugin_manager:
        column_config_type = plugin_manager.update_type_union(column_config_type, PluginType.COLUMN_GENERATOR)
    return column_config_type
