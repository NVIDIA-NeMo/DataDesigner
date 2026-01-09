# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.plugins.plugin import Plugin, PluginType

column_generator_plugin = Plugin(
    config_qualified_name="data_designer_e2e_tests.plugins.column_generator.config.DemoColumnGeneratorConfig",
    impl_qualified_name="data_designer_e2e_tests.plugins.column_generator.impl.DemoColumnGeneratorImpl",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
