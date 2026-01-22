# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

from data_designer.plugins.plugin import PluginType
from data_designer.plugins.registry import PluginRegistry

if TYPE_CHECKING:
    from data_designer.plugins.plugin import Plugin


class PluginManager:
    def __init__(self):
        self._plugin_registry = PluginRegistry()

    def get_column_generator_plugins(self) -> list[Plugin]:
        """Get all column generator plugins.

        Returns:
            A list of all column generator plugins.
        """
        return self._plugin_registry.get_plugins(PluginType.COLUMN_GENERATOR)

    def get_column_generator_plugin_if_exists(self, plugin_name: str) -> Plugin | None:
        """Get a column generator plugin by name if it exists.

        Args:
            plugin_name: The name of the plugin to retrieve.

        Returns:
            The plugin if found, otherwise None.
        """
        if self._plugin_registry.plugin_exists(plugin_name):
            return self._plugin_registry.get_plugin(plugin_name)

    def get_plugin_column_types(self, enum_type: type[Enum]) -> list[Enum]:
        """Get a list of plugin column types.

        Args:
            enum_type: The enum type to use for plugin entries.

        Returns:
            A list of plugin column types.
        """
        type_list = []
        for plugin in self._plugin_registry.get_plugins(PluginType.COLUMN_GENERATOR):
            type_list.append(enum_type(plugin.name))
        return type_list

    def inject_into_column_config_type_union(self, column_config_type: type[TypeAlias]) -> type[TypeAlias]:
        """Inject plugins into the column config type.

        Args:
            column_config_type: The column config type to inject plugins into.

        Returns:
            The column config type with plugins injected.
        """
        column_config_type = self._plugin_registry.add_plugin_types_to_union(
            column_config_type, PluginType.COLUMN_GENERATOR
        )
        return column_config_type

    def inject_into_seed_source_type_union(self, seed_source_type: type[TypeAlias]) -> type[TypeAlias]:
        """Inject plugins into the seed source type.

        Args:
            seed_source_type: The seed source type to inject plugins into.

        Returns:
            The seed source type with plugins injected.
        """
        seed_source_type = self._plugin_registry.add_plugin_types_to_union(seed_source_type, PluginType.SEED_READER)
        return seed_source_type
