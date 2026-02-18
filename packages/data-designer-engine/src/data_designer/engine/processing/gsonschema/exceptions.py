# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import data_designer.lazy_heavy_imports as lazy


class JSONSchemaValidationError(lazy.jsonschema.ValidationError):
    """Alias of ValidationError to ease imports."""
