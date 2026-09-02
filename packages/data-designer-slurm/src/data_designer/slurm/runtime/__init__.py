# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allocation-local runtime primitives for resolved Slurm plans."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_designer.slurm.runtime.bundle import stage_runtime_bundle  # noqa: F401
    from data_designer.slurm.runtime.controller import AllocationController, OneNodeAllocationController  # noqa: F401
    from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode  # noqa: F401

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AllocationController": (
        "data_designer.slurm.runtime.controller",
        "AllocationController",
    ),
    "OneNodeAllocationController": (
        "data_designer.slurm.runtime.controller",
        "OneNodeAllocationController",
    ),
    "SlurmRuntimeError": ("data_designer.slurm.runtime.errors", "SlurmRuntimeError"),
    "SlurmRuntimeErrorCode": ("data_designer.slurm.runtime.errors", "SlurmRuntimeErrorCode"),
    "stage_runtime_bundle": ("data_designer.slurm.runtime.bundle", "stage_runtime_bundle"),
}

__all__ = [
    "AllocationController",
    "OneNodeAllocationController",
    "SlurmRuntimeError",
    "SlurmRuntimeErrorCode",
    "stage_runtime_bundle",
]


def __getattr__(name: str) -> object:
    """Lazily load orchestration dependencies only when requested."""
    if name in _LAZY_IMPORTS:
        module_name, attribute_name = _LAZY_IMPORTS[name]
        attribute = getattr(importlib.import_module(module_name), attribute_name)
        globals()[name] = attribute
        return attribute
    raise AttributeError(f"module 'data_designer.slurm.runtime' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return public runtime exports for interactive discovery."""
    return __all__
