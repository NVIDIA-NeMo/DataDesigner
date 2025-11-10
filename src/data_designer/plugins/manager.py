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


class PluginManager:
    _instance = None
    _plugins_discovered = False
    _lock = threading.Lock()

    def __init__(self):
        self.registry = _PluginRegistry()
        if not self._plugins_discovered:
            self.discover()
            self._plugins_discovered = True

    def get_plugin(self, plugin_name: str) -> Plugin:
        return self.registry.get(plugin_name)

    def get_plugins(self, plugin_type: PluginType) -> list[Plugin]:
        return [plugin for plugin in self.registry._plugins.values() if plugin.plugin_type == plugin_type]

    def get_plugin_names(self, plugin_type: PluginType) -> list[str]:
        return [plugin.name for plugin in self.get_plugins(plugin_type)]

    def num_plugins(self, plugin_type: PluginType) -> int:
        return len(self.get_plugins(plugin_type))

    def update_type_union(self, type_union: Type[TypeAlias], plugin_type: PluginType) -> Type[TypeAlias]:
        for plugin in self.get_plugins(plugin_type):
            type_union |= plugin.config_cls
        return type_union

    def discover(self) -> Self:
        if PLUGINS_DISABLED:
            return self
        for ep in entry_points(group="data_designer.plugins"):
            try:
                plugin = ep.load()
                if isinstance(plugin, Plugin):
                    with self._lock:
                        self.registry.register_plugin(plugin)
                    logger.info(
                        f"🔌 Plugin discovered ➜ {plugin.plugin_type.value.replace('-', ' ')} "
                        f"{plugin.name.upper().replace('-', '_')} is now available ⚡️"
                    )
            except Exception as e:
                logger.warning(f"Failed to load plugin from entry point '{ep.name}': {e}")

        return self

    def __new__(cls, *args, **kwargs):
        """Plugin manager is a singleton."""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance


class _PluginRegistry:
    _plugins: dict[str, Plugin] = {}

    def get(self, plugin_name: str) -> Plugin:
        if plugin_name not in self._plugins:
            raise PluginNotFoundError(f"Plugin '{plugin_name}' not found.")
        return self._plugins[plugin_name]

    def register_plugin(self, plugin: Plugin) -> None:
        if plugin.name in self._plugins:
            raise PluginRegistrationError(f"Plugin '{plugin.name}' already registered.")
        self._plugins[plugin.name] = plugin

    def clear(self) -> None:
        """Clear all registered plugins. Primarily for testing purposes."""
        self._plugins.clear()
