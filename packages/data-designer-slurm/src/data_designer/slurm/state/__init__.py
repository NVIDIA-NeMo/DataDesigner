# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public state contracts for the optional Slurm execution package."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from data_designer.slurm.contracts import (
    ArtifactReference,
    AttemptId,
    ContractRecord,
    ContractValue,
    Identifier,
    RecordRange,
    ResumeWorkspace,
    Sha256Digest,
    ShardId,
)
from data_designer.slurm.state.artifacts import compute_candidate_schema_digest
from data_designer.slurm.state.base import (
    SchedulerIdentity,
    SchedulerJobIdentity,
    StateRecord,
    StateValue,
)
from data_designer.slurm.state.collection_records import (
    CollectedOutputFile,
    CollectionResult,
    CollectionState,
    CollectionStatus,
)
from data_designer.slurm.state.errors import (
    SlurmStateError,
    StateConflictError,
    StateCorruptionError,
    StateNotFoundError,
)
from data_designer.slurm.state.execution import (
    AttemptLifecycleState,
    AttemptManifest,
    AttemptTerminalClassification,
    RunManifest,
    ShardManifest,
)
from data_designer.slurm.state.observation import (
    SchedulerAccountingRecord,
    SchedulerObservationClient,
    SchedulerObservationCollector,
    SchedulerQueueRecord,
)
from data_designer.slurm.state.outputs import (
    CANDIDATE_OUTPUT_FORMAT,
    MAXIMUM_CANDIDATE_OUTPUT_FILES,
    CandidateOutcome,
    CandidateOutputFile,
    CandidateOutputManifest,
    CollectionPlan,
    CollectionShard,
    RetryPlan,
    RetryShard,
    ShardWinner,
)
from data_designer.slurm.state.readiness import (
    AttemptReadiness,
    DeploymentReadiness,
    EndpointPublicationState,
    ProbeEvidence,
    ProbeOutcome,
    ReadinessState,
    ReasonCode,
)
from data_designer.slurm.state.reconciliation import (
    reconcile_attempt_observation,
    validate_readiness_transition,
)
from data_designer.slurm.state.retry_records import RetryState, RetryStatus
from data_designer.slurm.state.scheduler import (
    EffectiveAttemptState,
    SchedulerObservation,
    SchedulerState,
)
from data_designer.slurm.state.status import (
    AttemptStatus,
    EffectiveRunState,
    GenerationState,
    RunStatus,
    ShardStatus,
)
from data_designer.slurm.state.validation import (
    StateContractError,
    validate_attempt_manifest,
    validate_attempt_set,
    validate_attempt_transition,
    validate_collection_plan,
    validate_scheduler_observation_transition,
    validate_shard_attempt_set,
    validate_shard_manifest,
    validate_shard_set,
    validate_shard_winner,
)

if TYPE_CHECKING:
    from data_designer.slurm.state.collection import SlurmCollectionCoordinator  # noqa: F401
    from data_designer.slurm.state.observer import SlurmStateReconciler  # noqa: F401
    from data_designer.slurm.state.retry import SlurmRetryCoordinator  # noqa: F401
    from data_designer.slurm.state.store import SlurmStateWriter  # noqa: F401

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "SlurmCollectionCoordinator": ("data_designer.slurm.state.collection", "SlurmCollectionCoordinator"),
    "SlurmStateReconciler": ("data_designer.slurm.state.observer", "SlurmStateReconciler"),
    "SlurmRetryCoordinator": ("data_designer.slurm.state.retry", "SlurmRetryCoordinator"),
    "SlurmStateWriter": ("data_designer.slurm.state.store", "SlurmStateWriter"),
}

__all__ = [
    "ArtifactReference",
    "AttemptLifecycleState",
    "AttemptManifest",
    "AttemptId",
    "AttemptReadiness",
    "AttemptStatus",
    "AttemptTerminalClassification",
    "CandidateOutcome",
    "CANDIDATE_OUTPUT_FORMAT",
    "MAXIMUM_CANDIDATE_OUTPUT_FILES",
    "CandidateOutputFile",
    "CandidateOutputManifest",
    "CollectedOutputFile",
    "compute_candidate_schema_digest",
    "CollectionPlan",
    "CollectionResult",
    "CollectionShard",
    "CollectionState",
    "CollectionStatus",
    "ContractRecord",
    "ContractValue",
    "DeploymentReadiness",
    "EffectiveAttemptState",
    "EffectiveRunState",
    "EndpointPublicationState",
    "Identifier",
    "GenerationState",
    "ProbeEvidence",
    "ProbeOutcome",
    "ReadinessState",
    "ReasonCode",
    "RecordRange",
    "RetryPlan",
    "RetryShard",
    "RetryState",
    "RetryStatus",
    "RunManifest",
    "RunStatus",
    "ResumeWorkspace",
    "SchedulerIdentity",
    "SchedulerJobIdentity",
    "SchedulerAccountingRecord",
    "SchedulerObservationClient",
    "SchedulerObservationCollector",
    "SchedulerQueueRecord",
    "SchedulerObservation",
    "SchedulerState",
    "Sha256Digest",
    "ShardManifest",
    "ShardStatus",
    "ShardId",
    "ShardWinner",
    "SlurmStateError",
    "SlurmCollectionCoordinator",
    "SlurmRetryCoordinator",
    "SlurmStateReconciler",
    "SlurmStateWriter",
    "StateConflictError",
    "StateContractError",
    "StateCorruptionError",
    "StateNotFoundError",
    "StateRecord",
    "StateValue",
    "reconcile_attempt_observation",
    "validate_attempt_manifest",
    "validate_attempt_set",
    "validate_attempt_transition",
    "validate_collection_plan",
    "validate_readiness_transition",
    "validate_scheduler_observation_transition",
    "validate_shard_attempt_set",
    "validate_shard_manifest",
    "validate_shard_set",
    "validate_shard_winner",
]


def __getattr__(name: str) -> object:
    """Lazily import state persistence exports when accessed."""
    if name in _LAZY_IMPORTS:
        module_path, attribute_name = _LAZY_IMPORTS[name]
        attribute = getattr(importlib.import_module(module_path), attribute_name)
        globals()[name] = attribute
        return attribute
    raise AttributeError(f"module 'data_designer.slurm.state' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the public state exports for interactive discovery."""
    return __all__
