# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Lazy imports facade for heavy third-party dependencies.

This module provides a centralized facade that lazily imports heavy dependencies
only when accessed, significantly improving import performance.

Usage:
    from data_designer.lazy_imports import pd, np, faker, litellm

    df = pd.DataFrame(...)
    arr = np.array([1, 2, 3])
    fake = faker.Faker()
"""


def __getattr__(name: str) -> object:
    """Lazily import heavy third-party dependencies when accessed.

    This allows fast imports of data_designer while deferring loading of heavy
    libraries until they're actually needed.

    Supported imports:
        - pd: pandas module
        - np: numpy module
        - pq: pyarrow.parquet module
        - pa: pyarrow module
        - faker: faker module
        - litellm: litellm module
        - sqlfluff: sqlfluff module
        - httpx: httpx module
        - duckdb: duckdb module
        - nx: networkx module
        - scipy: scipy module
        - jsonschema: jsonschema module
    """
    if name == "pd":
        import pandas as pd

        return pd
    elif name == "np":
        import numpy as np

        return np
    elif name == "pq":
        import pyarrow.parquet as pq

        return pq
    elif name == "pa":
        import pyarrow as pa

        return pa
    elif name == "faker":
        import faker

        return faker
    elif name == "litellm":
        import litellm

        return litellm
    elif name == "sqlfluff":
        import sqlfluff

        return sqlfluff
    elif name == "httpx":
        import httpx

        return httpx
    elif name == "duckdb":
        import duckdb

        return duckdb
    elif name == "nx":
        import networkx as nx

        return nx
    elif name == "scipy":
        import scipy

        return scipy
    elif name == "jsonschema":
        import jsonschema

        return jsonschema

    raise AttributeError(f"module 'data_designer.lazy_imports' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return list of available lazy imports."""
    return ["pd", "np", "pq", "pa", "faker", "litellm", "sqlfluff", "httpx", "duckdb", "nx", "scipy", "jsonschema"]
