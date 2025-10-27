# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from jsonschema import ValidationError


class JSONSchemaValidationError(ValidationError):
    """Alias of ValidationError to ease imports."""

    ...
