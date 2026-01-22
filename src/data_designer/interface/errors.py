# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.errors import DataDesignerError


class DataDesignerProfilingError(DataDesignerError):
    """Raised for errors related to a Data Designer dataset profiling."""


class DataDesignerGenerationError(DataDesignerError):
    """Raised for errors related to a Data Designer dataset generation."""


class InvalidBufferValueError(DataDesignerError):
    """Raised for errors related to an invalid buffer value."""
