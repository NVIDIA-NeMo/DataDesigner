# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public service facades and their dependency contracts."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from data_designer.slurm.services.benchmark import SlurmBenchmarkBackend, SlurmBenchmarkService
from data_designer.slurm.services.errors import (
    SlurmServiceError,
    SlurmServiceErrorCode,
    SlurmServiceOperation,
)
from data_designer.slurm.services.images import SlurmImageManager, SlurmImageResolver, SlurmImageService
from data_designer.slurm.services.results import (
    SlurmCollectionExecution,
    SlurmPersistedAttemptStatus,
    SlurmPersistedRunStatus,
    SlurmPersistedShardStatus,
    SlurmRetryExecution,
    SlurmRunCancellation,
    SlurmRunExecution,
)
from data_designer.slurm.services.run import SlurmBatchScriptRenderer, SlurmRunBackend, SlurmRunPlanner, SlurmRunService

if TYPE_CHECKING:
    from data_designer.slurm.services.wiring import (  # noqa: F401
        SlurmRunArtifactPublisher,
        create_slurm_image_service,
        create_slurm_run_service,
    )

_LAZY_IMPORTS = {
    "SlurmRunArtifactPublisher": ("data_designer.slurm.services.wiring", "SlurmRunArtifactPublisher"),
    "create_slurm_image_service": ("data_designer.slurm.services.wiring", "create_slurm_image_service"),
    "create_slurm_run_service": ("data_designer.slurm.services.wiring", "create_slurm_run_service"),
}

__all__ = [
    "SlurmBatchScriptRenderer",
    "SlurmBenchmarkBackend",
    "SlurmBenchmarkService",
    "SlurmCollectionExecution",
    "SlurmImageManager",
    "SlurmImageResolver",
    "SlurmImageService",
    "SlurmPersistedAttemptStatus",
    "SlurmPersistedRunStatus",
    "SlurmPersistedShardStatus",
    "SlurmRetryExecution",
    "SlurmRunArtifactPublisher",
    "SlurmRunBackend",
    "SlurmRunCancellation",
    "SlurmRunExecution",
    "SlurmRunPlanner",
    "SlurmRunService",
    "SlurmServiceError",
    "SlurmServiceErrorCode",
    "SlurmServiceOperation",
    "create_slurm_image_service",
    "create_slurm_run_service",
]


def __getattr__(name: str) -> object:
    """Lazily import production service wiring."""
    if name in _LAZY_IMPORTS:
        module_path, attribute_name = _LAZY_IMPORTS[name]
        attribute = getattr(importlib.import_module(module_path), attribute_name)
        globals()[name] = attribute
        return attribute
    raise AttributeError(f"module 'data_designer.slurm.services' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return public service exports."""
    return __all__
