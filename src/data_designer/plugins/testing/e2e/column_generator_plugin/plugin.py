# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.plugins.plugin import Plugin, PluginType

column_generator_plugin = Plugin(
    config_qualified_name="data_designer.plugins.testing.e2e.column_generator_plugin.config.TestColumnGeneratorConfig",
    impl_qualified_name="data_designer.plugins.testing.e2e.column_generator_plugin.impl.TestColumnGeneratorImpl",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
