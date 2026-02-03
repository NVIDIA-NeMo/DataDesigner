# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import threading
from importlib.metadata import entry_points
from typing import TypeAlias

from typing_extensions import Self

from data_designer.plugins.errors import PluginNotFoundError
from data_designer.plugins.plugin import Plugin, PluginType

logger = logging.getLogger(__name__)


PLUGINS_DISABLED = os.getenv("DISABLE_DATA_DESIGNER_PLUGINS", "false").lower() == "true"


class PluginRegistry:
    _instance = None
    _plugins_discovered = False
    _lock = threading.Lock()

    _plugins: dict[str, Plugin] = {}

    def __init__(self):
        with self._lock:
            if not self._plugins_discovered:
                self._discover()

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None
            cls._plugins_discovered = False
            cls._plugins = {}

    def add_plugin_types_to_union(self, type_union: type[TypeAlias], plugin_type: PluginType) -> type[TypeAlias]:
        for plugin in self.get_plugins(plugin_type):
            if plugin.config_cls not in type_union.__args__:
                type_union |= plugin.config_cls
        return type_union

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

    def _discover(self) -> Self:
        if PLUGINS_DISABLED:
            return self
        for ep in entry_points(group="data_designer.plugins"):
            try:
                plugin = ep.load()
                if isinstance(plugin, Plugin):
                    logger.info(
                        f"🔌 Plugin discovered ➜ {plugin.plugin_type.display_name} "
                        f"{plugin.enum_key_name} is now available ⚡️"
                    )
                    self._plugins[plugin.name] = plugin
            except Exception as e:
                logger.warning(f"🛑 Failed to load plugin from entry point {ep.name!r}: {e}")
        self._plugins_discovered = True
        return self

    def __new__(cls, *args, **kwargs):
        """Plugin manager is a singleton."""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
