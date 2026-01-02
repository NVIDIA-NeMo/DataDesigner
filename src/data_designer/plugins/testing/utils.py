# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.config.base import ConfigBase
from data_designer.engine.configurable_task import ConfigurableTask
from data_designer.plugins.plugin import Plugin


def assert_valid_plugin(plugin: Plugin) -> None:
    assert issubclass(plugin.config_cls, ConfigBase), "Plugin config class is not a subclass of ConfigBase"
    assert issubclass(plugin.task_cls, ConfigurableTask), "Plugin task class is not a subclass of ConfigurableTask"
