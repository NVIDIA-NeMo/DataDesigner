# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Lazy imports facade for heavy third-party dependencies.

This module provides a centralized facade that lazily imports heavy dependencies
only when accessed, significantly improving import performance.

The below mapping for lazy imports represents all third-party packages that, if used with
Data Designer, we strongly recommend using this lazy imports pattern to improve import performance.

Usage:
    from data_designer.lazy_heavy_imports import pd, np

    df = pd.DataFrame(...)
    arr = np.array([1, 2, 3])
"""

from __future__ import annotations

import importlib

# Mapping of lazy import names to their actual module paths
_LAZY_IMPORTS = {
    "pd": "pandas",
    "np": "numpy",
    "pq": "pyarrow.parquet",
    "pa": "pyarrow",
    "faker": "faker",
    "litellm": "litellm",
    "sqlfluff": "sqlfluff",
    "httpx": "httpx",
    "duckdb": "duckdb",
    "nx": "networkx",
    "scipy": "scipy",
    "jsonschema": "jsonschema",
}


def __getattr__(name: str) -> object:
    """Lazily import heavy third-party dependencies when accessed.

    This allows fast imports of data_designer while deferring loading of heavy
    libraries until they're actually needed.
    """
    if name in _LAZY_IMPORTS:
        module_name = _LAZY_IMPORTS[name]
        return importlib.import_module(module_name)

    raise AttributeError(f"module 'data_designer.lazy_heavy_imports' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return list of available lazy imports."""
    return list(_LAZY_IMPORTS.keys())
