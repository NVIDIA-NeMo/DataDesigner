import importlib.util
import inspect
import logging
import os
from pathlib import Path
import sys
import threading
from typing import Iterator, Optional, Type, TypeAlias

from typing_extensions import Self

from data_designer.plugins.errors import PluginNotFoundError, PluginRegistrationError
from data_designer.plugins.plugin import Plugin, PluginType

logger = logging.getLogger(__name__)


def _get_default_plugin_directory() -> Path:
    """Get the default plugin directory from environment or user's home directory.

    This function is called at runtime rather than at module import time,
    allowing tests to override the plugin directory via environment variables.
    """
    env_dir = os.getenv("DATA_DESIGNER_PLUGIN_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".data_designer" / "plugins"


class PluginManager:
    def __init__(self):
        self.registry = _PluginRegistry()

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

    def discover(self, plugin_dir: Optional[Path] = None) -> Self:
        plugin_dir = Path(plugin_dir or _get_default_plugin_directory())

        if not plugin_dir.exists():
            return self

        for file_path in plugin_dir.rglob("*.py"):
            if file_path.name.startswith("_"):
                continue

            for plugin in self._iter_plugins_from_file(file_path, plugin_dir):
                if isinstance(plugin, Plugin):
                    self.registry.register_plugin(plugin)
                    logger.info(
                        f"🔌 Plugin discovered ➜ {plugin.plugin_type.value.replace('-', ' ')} "
                        f"{plugin.name.upper().replace('-', '_')} is now available ⚡️"
                    )

        return self

    def _iter_plugins_from_file(self, file_path: Path, plugin_dir: Path) -> Optional[Iterator[Plugin]]:
        label = str(file_path.relative_to(plugin_dir)).replace("/", "_").replace(".", "_")
        module_name = f"_plugin_{label}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            for _, obj in inspect.getmembers(module):
                if isinstance(obj, Plugin):
                    yield obj

        except Exception:
            return


class _PluginRegistry:
    _plugins: dict[str, Plugin] = {}
    _instance = None
    _lock = threading.Lock()

    def get(self, plugin_name: str) -> Plugin:
        if plugin_name not in self._plugins:
            raise PluginNotFoundError(f"Plugin '{plugin_name}' not found.")
        return self._plugins[plugin_name]

    def register_plugin(self, plugin: Plugin) -> None:
        with self._lock:
            if plugin.name in self._plugins:
                raise PluginRegistrationError(f"Plugin '{plugin.name}' already registered.")
            self._plugins[plugin.name] = plugin

    def clear(self) -> None:
        """Clear all registered plugins. Primarily for testing purposes."""
        with self._lock:
            self._plugins.clear()

    def __new__(cls, *args, **kwargs):
        """Plugin registry is a singleton."""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
