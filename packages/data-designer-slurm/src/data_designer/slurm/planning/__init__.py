# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved execution-plan contracts for Data Designer Slurm."""

from __future__ import annotations

from data_designer.slurm.planning.compiler import (
    ConfigurationResolutionError,
    EffectiveDataDesignerSlurmConfig,
    PlanCompilationError,
    SlurmRunCompiler,
    compile_slurm_run_plan,
    resolve_slurm_config,
)
from data_designer.slurm.planning.models import (
    ArtifactReference,
    LockedPackage,
    PlannedShard,
    PortClaim,
    RecordRange,
    ResolvedBuilderInput,
    ResolvedClient,
    ResolvedDependencyLock,
    ResolvedDeployment,
    ResolvedImage,
    ResolvedInvocation,
    ResolvedOutput,
    ResolvedSlurmRunPlan,
    ResolvedSubmission,
    ResolvedTopology,
    ResumeWorkspace,
)
from data_designer.slurm.planning.validation import PlanContractError, validate_resolved_plan

__all__ = [
    "ArtifactReference",
    "ConfigurationResolutionError",
    "EffectiveDataDesignerSlurmConfig",
    "LockedPackage",
    "PlanContractError",
    "PlanCompilationError",
    "PlannedShard",
    "PortClaim",
    "RecordRange",
    "ResolvedBuilderInput",
    "ResolvedClient",
    "ResolvedDependencyLock",
    "ResolvedDeployment",
    "ResolvedImage",
    "ResolvedInvocation",
    "ResolvedOutput",
    "ResolvedSlurmRunPlan",
    "ResolvedSubmission",
    "ResolvedTopology",
    "ResumeWorkspace",
    "SlurmRunCompiler",
    "compile_slurm_run_plan",
    "resolve_slurm_config",
    "validate_resolved_plan",
]
