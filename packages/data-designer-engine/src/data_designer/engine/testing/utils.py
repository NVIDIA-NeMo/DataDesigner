# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.base import ConfigBase
from data_designer.engine.configurable_task import ConfigurableTask
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.resources.seed_reader import SeedReader
from data_designer.plugins.plugin import Plugin, PluginType

_PLUGIN_IMPLEMENTATION_BASES: dict[PluginType, type[object]] = {
    PluginType.COLUMN_GENERATOR: ConfigurableTask,
    PluginType.SEED_READER: SeedReader,
    PluginType.PROCESSOR: Processor,
}
if set(_PLUGIN_IMPLEMENTATION_BASES) != set(PluginType):
    raise AssertionError("Plugin implementation base map must cover all plugin types")


def _assert_subclass(cls: type[object], base_cls: type[object], message: str) -> None:
    if not issubclass(cls, base_cls):
        raise AssertionError(message)


def assert_valid_plugin(plugin: Plugin) -> None:
    _assert_subclass(plugin.config_cls, ConfigBase, "Plugin config class is not a subclass of ConfigBase")

    implementation_base = _PLUGIN_IMPLEMENTATION_BASES[plugin.plugin_type]
    _assert_subclass(
        plugin.impl_cls,
        implementation_base,
        f"{plugin.plugin_type.display_name.capitalize()} plugin impl class must be a subclass of "
        f"{implementation_base.__name__}",
    )
