# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from importlib.metadata import entry_points
import logging
import os
import threading
from typing import Type, TypeAlias

from typing_extensions import Self

from data_designer.plugins.errors import PluginNotFoundError, PluginRegistrationError
from data_designer.plugins.plugin import Plugin, PluginType

logger = logging.getLogger(__name__)


PLUGINS_DISABLED = os.getenv("DISABLE_DATA_DESIGNER_PLUGINS", "false").lower() == "true"


class PluginRegistry:
    _instance = None
    _plugins_discovered = False
    _lock = threading.Lock()

    _plugins: dict[str, Plugin] = {}

    def __init__(self):
        if not self._plugins_discovered:
            self.discover()
            self._plugins_discovered = True

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
        cls._plugins_discovered = False
        cls._plugins = {}

    def add_plugin(self, plugin: Plugin) -> None:
        if plugin.name in self._plugins:
            raise PluginRegistrationError(f"Plugin {plugin.name!r} already added.")
        self._plugins[plugin.name] = plugin

    def add_plugin_types(self, type_union: Type[TypeAlias], plugin_type: PluginType) -> Type[TypeAlias]:
        for plugin in self.get_plugins(plugin_type):
            type_union |= plugin.config_cls
        return type_union

    def clear_plugins(self) -> None:
        self._plugins.clear()

    def copy_plugins(self) -> dict[str, Plugin]:
        return deepcopy(self._plugins)

    def get_plugin(self, plugin_name: str) -> Plugin:
        if plugin_name not in self._plugins:
            raise PluginNotFoundError(f"Plugin {plugin_name!r} not found.")
        return self._plugins[plugin_name]

    def get_plugins(self, plugin_type: PluginType) -> list[Plugin]:
        return [plugin for plugin in self._plugins.values() if plugin.plugin_type == plugin_type]

    def get_plugin_names(self, plugin_type: PluginType) -> list[str]:
        return [plugin.name for plugin in self.get_plugins(plugin_type)]

    def num_plugins(self, plugin_type: PluginType) -> int:
        return len(self.get_plugins(plugin_type))

    def plugin_exists(self, plugin_name: str) -> bool:
        return plugin_name in self._plugins

    def set_plugins(self, plugins: dict[str, Plugin]) -> None:
        self._plugins = plugins

    def discover(self) -> Self:
        if PLUGINS_DISABLED:
            return self
        for ep in entry_points(group="data_designer.plugins"):
            try:
                plugin = ep.load()
                if isinstance(plugin, Plugin):
                    with self._lock:
                        self.add_plugin(plugin)
                    logger.info(
                        f"🔌 Plugin discovered ➜ {plugin.plugin_type.value.replace('-', ' ')} "
                        f"{plugin.enum_key_name} is now available ⚡️"
                    )
            except Exception as e:
                logger.warning(f"🛑 Failed to load plugin from entry point {ep.name!r}: {e}")

        return self

    def __new__(cls, *args, **kwargs):
        """Plugin manager is a singleton."""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
