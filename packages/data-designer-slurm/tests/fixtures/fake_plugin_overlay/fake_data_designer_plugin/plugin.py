# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from data_designer.config.base import SingleColumnConfig
from data_designer.plugins import Plugin, PluginType


class FakePluginConfig(SingleColumnConfig):
    """Minimal custom column configuration."""

    column_type: Literal["fake-slurm-column"] = "fake-slurm-column"


class FakePluginImplementation:
    """Minimal loadable implementation for entry-point verification."""

    def generate(self, data: dict[str, object]) -> dict[str, object]:
        """Return the provided record unchanged."""
        return data


plugin = Plugin(
    config_qualified_name="fake_data_designer_plugin.plugin.FakePluginConfig",
    impl_qualified_name="fake_data_designer_plugin.plugin.FakePluginImplementation",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
