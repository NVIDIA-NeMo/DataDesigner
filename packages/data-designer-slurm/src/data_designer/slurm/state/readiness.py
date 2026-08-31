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
        if self.ready_backends > self.expected_backends:
            raise ValueError("ready_backends must not exceed expected_backends")
        if self.endpoint_publication is EndpointPublicationState.FAILED and self.state not in {
            ReadinessState.FAILED,
            ReadinessState.STOPPED,
        }:
            raise ValueError("failed endpoint publication requires a failed or stopped deployment")

        if self.state is ReadinessState.PENDING:
            if self.ready_backends != 0:
                raise ValueError("pending deployments cannot have ready backends")
            if self.endpoint_publication is not EndpointPublicationState.PENDING:
                raise ValueError("pending deployments require pending endpoint publication")
        elif self.state is ReadinessState.STARTING:
            if (
                self.ready_backends == self.expected_backends
                and self.endpoint_publication is EndpointPublicationState.PUBLISHED
            ):
                raise ValueError("fully ready published deployments must use the ready state")
        elif self.state is ReadinessState.READY:
            if self.ready_backends != self.expected_backends:
                raise ValueError("ready deployments require every expected backend")
            if self.endpoint_publication is not EndpointPublicationState.PUBLISHED:
                raise ValueError("ready deployments require a published endpoint")
        elif self.state is ReadinessState.STOPPED and self.ready_backends != 0:
            raise ValueError("stopped deployments cannot have ready backends")
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
        deployment_ids = [deployment.deployment_id for deployment in self.deployments]
        model_aliases = [deployment.model_alias for deployment in self.deployments]
        if len(deployment_ids) != len(set(deployment_ids)):
            raise ValueError("deployment IDs must be unique")
        if len(model_aliases) != len(set(model_aliases)):
            raise ValueError("model aliases must be unique")
        if any(
            deployment.last_probe is not None and deployment.last_probe.observed_at > self.updated_at
            for deployment in self.deployments
        ):
            raise ValueError("probe observations must not be later than the readiness snapshot")

        deployment_states = tuple(deployment.state for deployment in self.deployments)
        if self.state is ReadinessState.PENDING:
            if any(state is not ReadinessState.PENDING for state in deployment_states):
                raise ValueError("pending attempts require every deployment to be pending")
        elif self.state is ReadinessState.STARTING:
            if any(state in {ReadinessState.FAILED, ReadinessState.STOPPED} for state in deployment_states):
                raise ValueError("starting attempts cannot contain failed or stopped deployments")
            if all(state is ReadinessState.READY for state in deployment_states):
                raise ValueError("attempts with every deployment ready must use the ready state")
        elif self.state is ReadinessState.READY:
            if any(state is not ReadinessState.READY for state in deployment_states):
                raise ValueError("ready attempts require every deployment to be ready")
        elif self.state is ReadinessState.FAILED:
            if not any(state is ReadinessState.FAILED for state in deployment_states):
                raise ValueError("failed attempts require at least one failed deployment")
        elif any(state is not ReadinessState.STOPPED for state in deployment_states):
            raise ValueError("stopped attempts require every deployment to be stopped")
        return self
