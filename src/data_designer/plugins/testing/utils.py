# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.config.base import ConfigBase
from data_designer.engine.configurable_task import ConfigurableTask
from data_designer.plugins.plugin import Plugin


def is_valid_plugin(plugin: Plugin) -> bool:
    if not isinstance(plugin.config_cls, ConfigBase):
        return False
    if not isinstance(plugin.task_cls, ConfigurableTask):
        return False

    return True
