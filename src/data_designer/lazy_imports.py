# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Lazy imports facade for heavy third-party dependencies.

This module provides a centralized facade that lazily imports heavy dependencies
(pandas, pyarrow, etc.) only when accessed, significantly improving import performance.

Usage:
    from data_designer import lazy_imports

    df = lazy_imports.pd.DataFrame(...)
    schema = lazy_imports.pq.read_schema(...)
"""


def __getattr__(name: str) -> object:
    """Lazily import heavy third-party dependencies when accessed.

    This allows fast imports of data_designer while deferring loading of heavy
    libraries like pandas and pyarrow until they're actually needed.

    Supported imports:
        - pd: pandas module
        - pq: pyarrow.parquet module
    """
    if name == "pd":
        import pandas as pd

        return pd
    elif name == "pq":
        import pyarrow.parquet as pq

        return pq

    raise AttributeError(f"module 'data_designer.lazy_imports' has no attribute {name!r}")


# For type checking
def __dir__() -> list[str]:
    """Return list of available lazy imports."""
    return ["pd", "pq"]
