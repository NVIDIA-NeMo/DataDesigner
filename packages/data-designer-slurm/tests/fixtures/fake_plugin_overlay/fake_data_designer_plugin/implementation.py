# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from data_designer.engine.column_generators.generators.base import ColumnGeneratorCellByCell
from fake_data_designer_plugin.plugin import FakePluginConfig


class FakePluginImplementation(ColumnGeneratorCellByCell[FakePluginConfig]):
    """Add a deterministic marker through the real generator contract."""

    def generate(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return the complete record with the plugin-owned column."""
        return {**data, self.config.name: "plugin-marker"}
