# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from data_designer.lazy_heavy_imports import jsonschema

if TYPE_CHECKING:
    import jsonschema


class JSONSchemaValidationError(jsonschema.ValidationError):
    """Alias of ValidationError to ease imports."""
