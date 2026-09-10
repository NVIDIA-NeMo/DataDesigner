# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from data_designer.config.base import SingleColumnConfig
from data_designer.plugins import Plugin, PluginType


class FakePluginConfig(SingleColumnConfig):
    """Minimal custom column configuration."""

    column_type: Literal["fake-slurm-column"] = "fake-slurm-column"
    model_alias: str | None = None
    judge_model_alias: str | None = None

    @property
    def required_columns(self) -> list[str]:
        return []

    @property
    def side_effect_columns(self) -> list[str]:
        return []

    def get_model_aliases(self) -> list[str]:
        return [alias for alias in (self.model_alias, self.judge_model_alias) if alias is not None]


plugin = Plugin(
    config_qualified_name="fake_data_designer_plugin.plugin.FakePluginConfig",
    impl_qualified_name="fake_data_designer_plugin.implementation.FakePluginImplementation",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
