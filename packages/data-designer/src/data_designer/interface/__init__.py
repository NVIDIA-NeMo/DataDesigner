# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_designer.config.run_config import ResumeMode  # noqa: F401
    from data_designer.interface.composite_workflow import (  # noqa: F401
        CompositeWorkflow,
        CompositeWorkflowResults,
        SkippedStageResult,
        SkippedStageStatus,
    )
    from data_designer.interface.data_designer import DataDesigner  # noqa: F401
    from data_designer.interface.errors import (  # noqa: F401
        DataDesignerEarlyShutdownError,
        DataDesignerGenerationError,
        DataDesignerProfilingError,
        DataDesignerWorkflowError,
        RecordRetryExhaustedError,
    )
    from data_designer.interface.record_retry import RetryExhaustion, RetryUntil, SamplerRetryMode  # noqa: F401
    from data_designer.interface.results import DatasetCreationResults  # noqa: F401

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "CompositeWorkflow": ("data_designer.interface.composite_workflow", "CompositeWorkflow"),
    "CompositeWorkflowResults": ("data_designer.interface.composite_workflow", "CompositeWorkflowResults"),
    "RecordRetryExhaustedError": ("data_designer.interface.errors", "RecordRetryExhaustedError"),
    "DataDesigner": ("data_designer.interface.data_designer", "DataDesigner"),
    "DataDesignerEarlyShutdownError": ("data_designer.interface.errors", "DataDesignerEarlyShutdownError"),
    "DataDesignerGenerationError": ("data_designer.interface.errors", "DataDesignerGenerationError"),
    "DataDesignerProfilingError": ("data_designer.interface.errors", "DataDesignerProfilingError"),
    "DataDesignerWorkflowError": ("data_designer.interface.errors", "DataDesignerWorkflowError"),
    "DatasetCreationResults": ("data_designer.interface.results", "DatasetCreationResults"),
    "RetryExhaustion": ("data_designer.interface.record_retry", "RetryExhaustion"),
    "RetryUntil": ("data_designer.interface.record_retry", "RetryUntil"),
    "ResumeMode": ("data_designer.config.run_config", "ResumeMode"),
    "SamplerRetryMode": ("data_designer.interface.record_retry", "SamplerRetryMode"),
    "SkippedStageResult": ("data_designer.interface.composite_workflow", "SkippedStageResult"),
    "SkippedStageStatus": ("data_designer.interface.composite_workflow", "SkippedStageStatus"),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> object:
    """Lazily import interface exports when accessed."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        attr = getattr(module, attr_name)
        # Cache so subsequent accesses find a real attribute and skip __getattr__.
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'data_designer.interface' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return available exports for tab-completion."""
    return __all__
