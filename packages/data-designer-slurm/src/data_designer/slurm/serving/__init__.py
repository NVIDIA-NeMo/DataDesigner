# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure server resolution for Data Designer Slurm."""

from __future__ import annotations

from data_designer.slurm.serving.compatibility import (
    UnsupportedServingRuntimeError,
    VllmRuntimeCompatibility,
    resolve_vllm_compatibility,
)
from data_designer.slurm.serving.context import ServerResolutionContext
from data_designer.slurm.serving.deployment import ResolvedServerDeployment
from data_designer.slurm.serving.endpoints import (
    ResolvedBackendEndpoint,
    ResolvedLogicalEndpoint,
    ResolvedReadinessProbe,
)
from data_designer.slurm.serving.processes import (
    VllmLaunchPolicy,
    VllmProcessRole,
    VllmProcessSpec,
    VllmRendezvousSpec,
)
from data_designer.slurm.serving.resolver import ServerResolutionError, resolve_server

__all__ = [
    "ResolvedBackendEndpoint",
    "ResolvedLogicalEndpoint",
    "ResolvedReadinessProbe",
    "ResolvedServerDeployment",
    "ServerResolutionContext",
    "ServerResolutionError",
    "UnsupportedServingRuntimeError",
    "VllmLaunchPolicy",
    "VllmProcessRole",
    "VllmProcessSpec",
    "VllmRendezvousSpec",
    "VllmRuntimeCompatibility",
    "resolve_server",
    "resolve_vllm_compatibility",
]
