# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Type, TypeAlias

from .config.utils.misc import can_run_data_designer_locally

if TYPE_CHECKING:
    from data_designer.plugins.plugin import Plugin


if can_run_data_designer_locally():
    from data_designer.plugins.plugin import PluginType
    from data_designer.plugins.registry import PluginRegistry


class PluginManager:
    def __init__(self):
        if can_run_data_designer_locally():
            self._plugins_available = True
            self._plugin_registry = PluginRegistry()
        else:
            self._plugins_available = False
            self._plugin_registry = None

    def get_plugin_column_configs(self) -> list[Plugin]:
        """Get all plugin column configs.

        Returns:
            A list of all plugin column configs.
        """
        if self._plugins_available:
            return [
                self._plugin_registry.get_plugin(plugin_name)
                for plugin_name in self._plugin_registry.get_plugin_names(PluginType.COLUMN_GENERATOR)
            ]
        return []

    def get_plugin_column_config_if_available(self, plugin_name: str) -> Plugin | None:
        """Get a plugin column config by name if available.

        Args:
            plugin_name: The name of the plugin to retrieve.

        Returns:
            The plugin if found, otherwise None.
        """
        if self._plugins_available:
            for name in self._plugin_registry.get_plugin_names(PluginType.COLUMN_GENERATOR):
                if plugin_name == name:
                    return self._plugin_registry.get_plugin(plugin_name)
        return None

    def get_plugin_column_types(self, enum_type: Type[Enum], required_resources: list[str] | None = None) -> list[Enum]:
        """Get a list of plugin column types.

        Args:
            enum_type: The enum type to use for plugin entries.
            required_resources: If provided, only return plugins with the required resources.

        Returns:
            A list of plugin column types.
        """
        type_list = []
        if self._plugins_available:
            for plugin in self._plugin_registry.get_plugins(PluginType.COLUMN_GENERATOR):
                if required_resources:
                    task_required_resources = plugin.task_cls.metadata().required_resources or []
                    if not all(resource in task_required_resources for resource in required_resources):
                        continue
                type_list.append(enum_type(plugin.name))
        return type_list

    def inject_into_column_config_type_union(self, column_config_type: Type[TypeAlias]) -> Type[TypeAlias]:
        """Inject plugins into the column config type.

        Args:
            column_config_type: The column config type to inject plugins into.

        Returns:
            The column config type with plugins injected.
        """
        if self._plugins_available:
            column_config_type = self._plugin_registry.add_plugin_types(column_config_type, PluginType.COLUMN_GENERATOR)
        return column_config_type
