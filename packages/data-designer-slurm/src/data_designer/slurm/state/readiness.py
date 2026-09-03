# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import Field, NonNegativeInt, PositiveInt, StringConstraints, field_validator, model_validator

from data_designer.slurm.contracts import AttemptId, Identifier, ShardId
from data_designer.slurm.state.base import StateRecord, StateValue, validate_utc_timestamp

ReasonCode = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    ),
]


class ReadinessState(str, Enum):
    PENDING = "pending"
    RESTARTING = "restarting"
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    STOPPED = "stopped"


class EndpointPublicationState(str, Enum):
    PENDING = "pending"
    PUBLISHED = "published"
    FAILED = "failed"


class ProbeOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class ProbeEvidence(StateValue):
    """Bounded, redacted evidence from one readiness probe."""

    observed_at: datetime
    outcome: ProbeOutcome
    reason_code: ReasonCode
    redacted_message: Annotated[str, StringConstraints(max_length=512)]

    _observed_at_is_utc = field_validator("observed_at")(validate_utc_timestamp)

    @field_validator("redacted_message")
    @classmethod
    def validate_redacted_message(cls, value: str) -> str:
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("redacted_message must not contain control characters")
        return value


class DeploymentReadiness(StateValue):
    """Readiness state for one authored-order model deployment."""

    deployment_id: Identifier
    model_alias: str
    state: ReadinessState
    expected_backends: PositiveInt
    ready_backends: NonNegativeInt
    endpoint_publication: EndpointPublicationState
    last_probe: ProbeEvidence | None = None

    @model_validator(mode="after")
    def validate_counts_and_state(self) -> DeploymentReadiness:
        _validate_deployment_common(self)
        _validate_deployment_state(self)
        return self


class AttemptReadiness(StateRecord):
    """Revisioned readiness snapshot for all deployments in one attempt."""

    run_id: Identifier
    shard_id: ShardId
    attempt_id: AttemptId
    revision: PositiveInt
    updated_at: datetime
    state: ReadinessState
    deployments: tuple[DeploymentReadiness, ...] = Field(min_length=1)

    _updated_at_is_utc = field_validator("updated_at")(validate_utc_timestamp)

    @model_validator(mode="after")
    def validate_deployments(self) -> AttemptReadiness:
        _validate_deployment_uniqueness(self.deployments)
        _validate_probe_chronology(self.deployments, self.updated_at)
        _validate_attempt_state(self.state, self.deployments)
        return self


def _validate_deployment_common(deployment: DeploymentReadiness) -> None:
    if deployment.ready_backends > deployment.expected_backends:
        raise ValueError("ready_backends must not exceed expected_backends")
    if deployment.endpoint_publication is EndpointPublicationState.FAILED and deployment.state not in {
        ReadinessState.FAILED,
        ReadinessState.STOPPED,
    }:
        raise ValueError("failed endpoint publication requires a failed or stopped deployment")


def _validate_deployment_state(deployment: DeploymentReadiness) -> None:
    if deployment.state in {ReadinessState.PENDING, ReadinessState.RESTARTING}:
        _validate_inactive_deployment(deployment)
    elif deployment.state is ReadinessState.STARTING:
        _validate_starting_deployment(deployment)
    elif deployment.state is ReadinessState.READY:
        _validate_ready_deployment(deployment)
    elif deployment.state is ReadinessState.STOPPED and deployment.ready_backends != 0:
        raise ValueError("stopped deployments cannot have ready backends")


def _validate_inactive_deployment(deployment: DeploymentReadiness) -> None:
    if deployment.ready_backends != 0:
        raise ValueError("pending or restarting deployments cannot have ready backends")
    if deployment.endpoint_publication is not EndpointPublicationState.PENDING:
        raise ValueError("pending or restarting deployments require pending endpoint publication")


def _validate_starting_deployment(deployment: DeploymentReadiness) -> None:
    if (
        deployment.ready_backends == deployment.expected_backends
        and deployment.endpoint_publication is EndpointPublicationState.PUBLISHED
    ):
        raise ValueError("fully ready published deployments must use the ready state")


def _validate_ready_deployment(deployment: DeploymentReadiness) -> None:
    if deployment.ready_backends != deployment.expected_backends:
        raise ValueError("ready deployments require every expected backend")
    if deployment.endpoint_publication is not EndpointPublicationState.PUBLISHED:
        raise ValueError("ready deployments require a published endpoint")


def _validate_deployment_uniqueness(deployments: tuple[DeploymentReadiness, ...]) -> None:
    deployment_ids = [deployment.deployment_id for deployment in deployments]
    model_aliases = [deployment.model_alias for deployment in deployments]
    if len(deployment_ids) != len(set(deployment_ids)):
        raise ValueError("deployment IDs must be unique")
    if len(model_aliases) != len(set(model_aliases)):
        raise ValueError("model aliases must be unique")


def _validate_probe_chronology(deployments: tuple[DeploymentReadiness, ...], updated_at: datetime) -> None:
    if any(
        deployment.last_probe is not None and deployment.last_probe.observed_at > updated_at
        for deployment in deployments
    ):
        raise ValueError("probe observations must not be later than the readiness snapshot")


def _validate_attempt_state(state: ReadinessState, deployments: tuple[DeploymentReadiness, ...]) -> None:
    deployment_states = tuple(deployment.state for deployment in deployments)
    if state in {ReadinessState.PENDING, ReadinessState.RESTARTING}:
        if any(deployment_state is not state for deployment_state in deployment_states):
            raise ValueError(f"{state.value} attempts require every deployment to be {state.value}")
    elif state is ReadinessState.STARTING:
        if any(item in {ReadinessState.FAILED, ReadinessState.STOPPED} for item in deployment_states):
            raise ValueError("starting attempts cannot contain failed or stopped deployments")
        if all(item is ReadinessState.READY for item in deployment_states):
            raise ValueError("attempts with every deployment ready must use the ready state")
    elif state is ReadinessState.READY and any(item is not ReadinessState.READY for item in deployment_states):
        raise ValueError("ready attempts require every deployment to be ready")
    elif state is ReadinessState.FAILED and not any(item is ReadinessState.FAILED for item in deployment_states):
        raise ValueError("failed attempts require at least one failed deployment")
    elif state is ReadinessState.STOPPED and any(item is not ReadinessState.STOPPED for item in deployment_states):
        raise ValueError("stopped attempts require every deployment to be stopped")
