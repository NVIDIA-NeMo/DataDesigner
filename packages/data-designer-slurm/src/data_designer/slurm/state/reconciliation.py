# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timedelta

from data_designer.slurm.state.execution import AttemptLifecycleState, AttemptManifest
from data_designer.slurm.state.readiness import (
    AttemptReadiness,
    EndpointPublicationState,
    ReadinessState,
)
from data_designer.slurm.state.scheduler import (
    EffectiveAttemptState,
    SchedulerObservation,
    SchedulerState,
)
from data_designer.slurm.state.validation import StateContractError

_ALLOWED_READINESS_TRANSITIONS: dict[ReadinessState, frozenset[ReadinessState]] = {
    ReadinessState.PENDING: frozenset(
        {ReadinessState.PENDING, ReadinessState.STARTING, ReadinessState.FAILED, ReadinessState.STOPPED}
    ),
    ReadinessState.STARTING: frozenset(
        {ReadinessState.STARTING, ReadinessState.READY, ReadinessState.FAILED, ReadinessState.STOPPED}
    ),
    ReadinessState.READY: frozenset({ReadinessState.READY, ReadinessState.FAILED, ReadinessState.STOPPED}),
    ReadinessState.FAILED: frozenset({ReadinessState.FAILED, ReadinessState.STOPPED}),
    ReadinessState.STOPPED: frozenset({ReadinessState.STOPPED}),
}

_ALLOWED_ENDPOINT_TRANSITIONS: dict[EndpointPublicationState, frozenset[EndpointPublicationState]] = {
    EndpointPublicationState.PENDING: frozenset(
        {
            EndpointPublicationState.PENDING,
            EndpointPublicationState.PUBLISHED,
            EndpointPublicationState.FAILED,
        }
    ),
    EndpointPublicationState.PUBLISHED: frozenset({EndpointPublicationState.PUBLISHED}),
    EndpointPublicationState.FAILED: frozenset({EndpointPublicationState.FAILED}),
}

_SCHEDULER_FAILURE_STATES = frozenset(
    {
        SchedulerState.FAILED,
        SchedulerState.CANCELLED,
        SchedulerState.TIMED_OUT,
        SchedulerState.NODE_FAILED,
        SchedulerState.PREEMPTED,
        SchedulerState.REQUEUED,
        SchedulerState.OUT_OF_MEMORY,
    }
)


def validate_readiness_transition(
    previous: AttemptReadiness,
    current: AttemptReadiness,
) -> AttemptReadiness:
    """Validate monotonic revision, identity, order, and readiness transitions."""
    _require(previous.run_id == current.run_id, "readiness run_id cannot change")
    _require(previous.shard_id == current.shard_id, "readiness shard_id cannot change")
    _require(previous.attempt_id == current.attempt_id, "readiness attempt_id cannot change")
    _require(current.revision == previous.revision + 1, "readiness revision must increase by exactly one")
    _require(current.updated_at >= previous.updated_at, "readiness updated_at cannot move backward")
    _require(
        current.state in _ALLOWED_READINESS_TRANSITIONS[previous.state],
        f"attempt readiness cannot move from {previous.state.value} to {current.state.value}",
    )
    _require(
        len(previous.deployments) == len(current.deployments),
        "readiness deployment count cannot change",
    )

    for old_deployment, new_deployment in zip(previous.deployments, current.deployments, strict=True):
        _require(
            old_deployment.deployment_name == new_deployment.deployment_name,
            "readiness deployment order or name cannot change",
        )
        _require(
            old_deployment.model_alias == new_deployment.model_alias,
            "readiness deployment model alias cannot change",
        )
        _require(
            old_deployment.expected_backends == new_deployment.expected_backends,
            "readiness expected backend count cannot change",
        )
        _require(
            new_deployment.state in _ALLOWED_READINESS_TRANSITIONS[old_deployment.state],
            (
                f"deployment {old_deployment.deployment_name!r} cannot move from "
                f"{old_deployment.state.value} to {new_deployment.state.value}"
            ),
        )
        _require(
            new_deployment.endpoint_publication in _ALLOWED_ENDPOINT_TRANSITIONS[old_deployment.endpoint_publication],
            f"deployment {old_deployment.deployment_name!r} endpoint publication cannot move backward",
        )
    return current


def reconcile_attempt_observation(
    attempt: AttemptManifest,
    readiness: AttemptReadiness,
    scheduler: SchedulerObservation,
    *,
    current_time: datetime,
) -> EffectiveAttemptState:
    """Apply scheduler terminal precedence without treating readiness as success."""
    _require_utc(current_time, "current_time")
    _require(current_time >= scheduler.observed_at, "current_time cannot precede scheduler observation")
    _require(readiness.run_id == attempt.run_id, "readiness run_id does not match attempt")
    _require(readiness.shard_id == attempt.shard_id, "readiness shard_id does not match attempt")
    _require(readiness.attempt_id == attempt.attempt_id, "readiness attempt_id does not match attempt")
    _require(attempt.scheduler is not None, "attempt has no scheduler identity")
    _require(scheduler.scheduler == attempt.scheduler, "scheduler identity does not match attempt")

    if scheduler.state in _SCHEDULER_FAILURE_STATES:
        return EffectiveAttemptState.FAILED
    if attempt.state is AttemptLifecycleState.FAILED:
        return EffectiveAttemptState.FAILED
    if attempt.state is AttemptLifecycleState.SUCCEEDED:
        return EffectiveAttemptState.SUCCEEDED

    if scheduler.state is SchedulerState.COMPLETED:
        return EffectiveAttemptState.FAILED
    if scheduler.state is SchedulerState.ACCOUNTING_LAG:
        deadline = scheduler.reconciliation_deadline
        _require(deadline is not None, "accounting lag has no reconciliation deadline")
        if current_time <= deadline:
            return EffectiveAttemptState.ACCOUNTING_LAG
        return EffectiveAttemptState.UNKNOWN
    if scheduler.state is SchedulerState.PENDING:
        return EffectiveAttemptState.PENDING
    if scheduler.state is SchedulerState.RUNNING:
        return EffectiveAttemptState.RUNNING

    if readiness.state is ReadinessState.FAILED:
        return EffectiveAttemptState.FAILED
    if readiness.state is ReadinessState.PENDING:
        return EffectiveAttemptState.PENDING
    if readiness.state in {ReadinessState.STARTING, ReadinessState.READY}:
        return EffectiveAttemptState.RUNNING
    return EffectiveAttemptState.UNKNOWN


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StateContractError(message)


def _require_utc(value: datetime, field_name: str) -> None:
    _require(
        value.tzinfo is not None and value.utcoffset() == timedelta(0),
        f"{field_name} must be in UTC",
    )
