# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.plugins.plugin import Plugin, PluginType

seed_reader_plugin = Plugin(
    config_qualified_name="data_designer_e2e_tests.plugins.seed_reader.config.DemoSeedSource",
    impl_qualified_name="data_designer_e2e_tests.plugins.seed_reader.impl.DemoSeedReader",
    plugin_type=PluginType.SEED_READER,
)
