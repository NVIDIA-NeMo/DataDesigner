# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.lazy_imports import jsonschema


class JSONSchemaValidationError(jsonschema.ValidationError):
    """Alias of ValidationError to ease imports."""
