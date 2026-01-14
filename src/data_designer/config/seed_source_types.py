# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated

from pydantic import Field
from typing_extensions import TypeAlias

from data_designer.config.seed_source import DataFrameSeedSource, HuggingFaceSeedSource, LocalFileSeedSource
from data_designer.plugin_manager import PluginManager

plugin_manager = PluginManager()

_SeedSourceT: TypeAlias = LocalFileSeedSource | HuggingFaceSeedSource | DataFrameSeedSource
_SeedSourceT = plugin_manager.inject_into_seed_source_type_union(_SeedSourceT)

SeedSourceT = Annotated[_SeedSourceT, Field(discriminator="seed_type")]
