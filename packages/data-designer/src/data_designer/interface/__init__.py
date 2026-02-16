# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_designer.interface.data_designer import DataDesigner  # noqa: F401
    from data_designer.interface.errors import (  # noqa: F401
        DataDesignerGenerationError,
        DataDesignerProfilingError,
    )
    from data_designer.interface.results import DatasetCreationResults  # noqa: F401

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DataDesigner": ("data_designer.interface.data_designer", "DataDesigner"),
    "DataDesignerGenerationError": ("data_designer.interface.errors", "DataDesignerGenerationError"),
    "DataDesignerProfilingError": ("data_designer.interface.errors", "DataDesignerProfilingError"),
    "DatasetCreationResults": ("data_designer.interface.results", "DatasetCreationResults"),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> object:
    """Lazily import interface exports when accessed."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module 'data_designer.interface' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return available exports for tab-completion."""
    return __all__
