# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, TypeVar

T_primitive = TypeVar("T_primitive", str, int, float, bool)
DataObjectT = dict | list | str | int | float | bool
JSONSchemaT = dict[str, Any]
