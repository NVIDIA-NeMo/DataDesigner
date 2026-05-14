# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata


def get_library_version() -> str:
    """Get the installed library version, or 'unknown' if not available."""
    try:
        return importlib.metadata.version("data-designer-config")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
