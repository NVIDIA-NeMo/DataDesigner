# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class ProviderModelKey:
    provider_name: str
    model_id: str


@dataclass
class ProviderModelStaticCap:
    cap: int
    aliases: tuple[str, ...]
    raw_caps: Mapping[str, int | None]
    merge_rule: str = "min_same_endpoint"
