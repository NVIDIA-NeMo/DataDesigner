# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.plugins.plugin import Plugin, PluginType

markdown_section_seed_reader_plugin = Plugin(
    config_qualified_name="data_designer_markdown_seed_reader.config.MarkdownSectionSeedSource",
    impl_qualified_name="data_designer_markdown_seed_reader.impl.MarkdownSectionSeedReader",
    plugin_type=PluginType.SEED_READER,
)
