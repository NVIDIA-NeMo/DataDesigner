# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public state contracts for the optional Slurm execution package."""

from __future__ import annotations

from data_designer.slurm.contracts import (
    ArtifactReference,
    AttemptId,
    ContractRecord,
    ContractValue,
    Identifier,
    ModelAlias,
    RecordRange,
    ResumeWorkspace,
    Sha256Digest,
    ShardId,
)
from data_designer.slurm.state.base import (
    SchedulerIdentity,
    StateRecord,
    StateValue,
)
from data_designer.slurm.state.execution import (
    AttemptLifecycleState,
    AttemptManifest,
    AttemptTerminalClassification,
    RunManifest,
    ShardManifest,
)
from data_designer.slurm.state.outputs import (
    CandidateOutcome,
    CandidateOutputFile,
    CandidateOutputManifest,
    CollectionPlan,
    CollectionShard,
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
from data_designer.slurm.state.scheduler import (
    EffectiveAttemptState,
    SchedulerObservation,
    SchedulerState,
)
from data_designer.slurm.state.validation import (
    StateContractError,
    validate_attempt_manifest,
    validate_attempt_set,
    validate_attempt_transition,
    validate_collection_plan,
    validate_scheduler_observation_transition,
    validate_shard_manifest,
    validate_shard_set,
    validate_shard_winner,
)

__all__ = [
    "ArtifactReference",
    "AttemptLifecycleState",
    "AttemptManifest",
    "AttemptId",
    "AttemptReadiness",
    "AttemptTerminalClassification",
    "CandidateOutcome",
    "CandidateOutputFile",
    "CandidateOutputManifest",
    "CollectionPlan",
    "CollectionShard",
    "ContractRecord",
    "ContractValue",
    "DeploymentReadiness",
    "EffectiveAttemptState",
    "EndpointPublicationState",
    "Identifier",
    "ModelAlias",
    "ProbeEvidence",
    "ProbeOutcome",
    "ReadinessState",
    "ReasonCode",
    "RecordRange",
    "RunManifest",
    "ResumeWorkspace",
    "SchedulerIdentity",
    "SchedulerObservation",
    "SchedulerState",
    "Sha256Digest",
    "ShardManifest",
    "ShardId",
    "ShardWinner",
    "StateContractError",
    "StateRecord",
    "StateValue",
    "reconcile_attempt_observation",
    "validate_attempt_manifest",
    "validate_attempt_set",
    "validate_attempt_transition",
    "validate_collection_plan",
    "validate_readiness_transition",
    "validate_scheduler_observation_transition",
    "validate_shard_manifest",
    "validate_shard_set",
    "validate_shard_winner",
]
