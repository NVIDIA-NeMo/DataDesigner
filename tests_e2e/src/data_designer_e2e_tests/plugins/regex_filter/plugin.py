# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.plugins.plugin import Plugin, PluginType

regex_filter_plugin = Plugin(
    config_qualified_name="data_designer_e2e_tests.plugins.regex_filter.config.RegexFilterProcessorConfig",
    impl_qualified_name="data_designer_e2e_tests.plugins.regex_filter.impl.RegexFilterProcessor",
    plugin_type=PluginType.PROCESSOR,
)
