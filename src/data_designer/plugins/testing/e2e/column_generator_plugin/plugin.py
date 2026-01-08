# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.plugins.plugin import Plugin, PluginType

plugin = Plugin(
    config_qualified_name=f"{__name__}.TestSeedSource",
    impl_qualified_name=f"{__name__}.TestSeedReader",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
