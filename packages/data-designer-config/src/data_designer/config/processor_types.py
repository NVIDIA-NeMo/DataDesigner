# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing_extensions import TypeAlias

from data_designer.config.processors import DropColumnsProcessorConfig, SchemaTransformProcessorConfig
from data_designer.plugin_manager import PluginManager

plugin_manager = PluginManager()

ProcessorConfigT: TypeAlias = DropColumnsProcessorConfig | SchemaTransformProcessorConfig
ProcessorConfigT = plugin_manager.inject_into_processor_config_type_union(ProcessorConfigT)
