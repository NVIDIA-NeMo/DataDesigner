# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.base import ConfigBase
from data_designer.engine.configurable_task import ConfigurableTask
from data_designer.engine.resources.seed_reader import SeedReader
from data_designer.plugins.plugin import Plugin, PluginType


def assert_valid_plugin(plugin: Plugin) -> None:
    assert issubclass(plugin.config_cls, ConfigBase), "Plugin config class is not a subclass of ConfigBase"

    if plugin.plugin_type == PluginType.COLUMN_GENERATOR:
        assert issubclass(plugin.impl_cls, ConfigurableTask), (
            "Column generator plugin impl class must be a subclass of ConfigurableTask"
        )
    elif plugin.plugin_type == PluginType.SEED_READER:
        assert issubclass(plugin.impl_cls, SeedReader), "Seed reader plugin impl class must be a subclass of SeedReader"
