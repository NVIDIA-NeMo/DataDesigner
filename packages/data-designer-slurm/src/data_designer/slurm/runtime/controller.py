# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-node allocation orchestration with one process and cleanup owner."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.runtime.models import AllocationContext, RuntimeEndpoint
from data_designer.slurm.runtime.preflight import AllocationPreflight
from data_designer.slurm.runtime.probes import ReadinessProber
from data_designer.slurm.runtime.records import load_complete_client_candidate
from data_designer.slurm.runtime.steps import (
    ClientStepBuilder,
    build_endpoint_steps,
    build_vllm_steps,
)
from data_designer.slurm.runtime.supervisor import ManagedStep, RuntimeClock, StepSupervisor
from data_designer.slurm.serving.deployment import ResolvedVllmServerDeployment
from data_designer.slurm.serving.resolver import resolve_vllm_server
from data_designer.slurm.state import (
    AttemptLifecycleState,
    AttemptManifest,
    AttemptReadiness,
    AttemptTerminalClassification,
    DeploymentReadiness,
    EndpointPublicationState,
    ProbeEvidence,
    ProbeOutcome,
    ReadinessState,
)


class RuntimeStateStore(Protocol):
    """State-writer operations required by one allocation."""

    def update_attempt(self, attempt: AttemptManifest) -> AttemptManifest:
        """Persist a monotonic attempt update."""
        ...

    def write_readiness(self, readiness: AttemptReadiness) -> AttemptReadiness:
        """Persist the exact next readiness revision."""
        ...


@dataclass(slots=True)
class _DeploymentStatus:
    deployment: ResolvedVllmServerDeployment
    state: ReadinessState = ReadinessState.PENDING
    ready_backends: int = 0
    endpoint_publication: EndpointPublicationState = EndpointPublicationState.PENDING
    last_probe: ProbeEvidence | None = None


class OneNodeAllocationController:
    """Execute one planned shard attempt and finalize only complete candidates."""

    def __init__(
        self,
        context: AllocationContext,
        *,
        runtime_proxy_path: Path,
        state: RuntimeStateStore,
        supervisor: StepSupervisor,
        preflight: AllocationPreflight,
        client_steps: ClientStepBuilder,
        prober: ReadinessProber,
        clock: RuntimeClock,
        environment: Mapping[str, str],
        poll_interval_seconds: float = 0.5,
        probe_timeout_seconds: float = 1.0,
    ) -> None:
        if poll_interval_seconds <= 0 or probe_timeout_seconds <= 0:
            raise ValueError("runtime probe intervals must be positive")
        self._context = context
        self._runtime_proxy_path = runtime_proxy_path
        self._state = state
        self._supervisor = supervisor
        self._preflight = preflight
        self._client_steps = client_steps
        self._prober = prober
        self._clock = clock
        self._environment = environment
        self._poll_interval_seconds = poll_interval_seconds
        self._probe_timeout_seconds = probe_timeout_seconds
        self._attempt = context.attempt
        self._readiness: AttemptReadiness | None = None
        self._statuses: list[_DeploymentStatus] = []

    def run(self) -> AttemptManifest:
        """Run the allocation and persist one terminal attempt on every owned path."""
        failure: SlurmRuntimeError | None = None
        failure_cause: BaseException | None = None
        candidate_reference: ArtifactReference | None = None
        client_completed_at: datetime | None = None
        try:
            candidate_reference, client_completed_at = self._execute()
        except BaseException as error:
            failure, failure_cause = _normalize_failure(error), error

        if failure is not None:
            failure = self._record_failed_readiness(failure)
        try:
            self._supervisor.cleanup()
        except BaseException as error:
            if failure is None:
                failure, failure_cause = _normalize_failure(error), error
                failure = self._record_failed_readiness(failure)
            else:
                failure.add_note(f"cleanup also failed: {type(error).__name__}")
        if self._readiness is not None:
            try:
                self._publish_stopped_readiness()
            except BaseException as error:
                if failure is None:
                    failure, failure_cause = _normalize_failure(error), error
                else:
                    failure.add_note(f"stopped readiness could not be persisted: {type(error).__name__}")
        try:
            terminal = self._publish_terminal_attempt(
                failure=failure,
                candidate_reference=candidate_reference,
                client_completed_at=client_completed_at,
            )
        except BaseException as error:
            terminal_failure = _normalize_failure(error)
            if failure is None:
                failure, failure_cause = terminal_failure, error
                try:
                    terminal = self._publish_terminal_attempt(
                        failure=failure,
                        candidate_reference=None,
                        client_completed_at=None,
                    )
                except BaseException as retry_error:
                    failure.add_note(f"failed terminal attempt could not be persisted: {type(retry_error).__name__}")
                    terminal = self._attempt
            else:
                failure.add_note(f"terminal attempt could not be persisted: {type(error).__name__}")
                terminal = self._attempt
        if failure is not None:
            if failure_cause is failure:
                raise failure
            raise failure from failure_cause
        return terminal

    def _execute(self) -> tuple[ArtifactReference, datetime]:
        self._preflight.verify(self._context, self._environment)
        self._validate_attempt_state()
        self._mark_attempt_running()
        deployments = tuple(
            resolve_vllm_server(self._context.plan, deployment.deployment_id)
            for deployment in self._context.plan.deployments
        )
        self._statuses = [_DeploymentStatus(deployment) for deployment in deployments]
        self._publish_readiness(ReadinessState.PENDING)

        endpoint_steps = build_endpoint_steps(
            deployments,
            self._context.plan,
            self._context.attempt_directory,
            self._environment,
            self._runtime_proxy_path,
        )
        endpoints = tuple(endpoint for _, endpoint in endpoint_steps)
        client_preflight = self._client_steps.build_preflight_step(
            self._context.plan,
            self._context.shard,
            self._attempt,
            self._context.attempt_directory,
            endpoints,
            self._environment,
        )
        self._supervisor.wait(self._supervisor.start(client_preflight))

        for status in self._statuses:
            status.state = ReadinessState.STARTING
        self._publish_readiness(ReadinessState.STARTING)
        server_processes = self._start_servers(deployments)
        for status, required in zip(self._statuses, server_processes, strict=True):
            self._wait_for_backends(status, required)
        self._publish_readiness(ReadinessState.STARTING)

        endpoint_processes = tuple(self._supervisor.start(step) for step, _ in endpoint_steps)
        required_processes = tuple(process for group in server_processes for process in group) + endpoint_processes
        self._wait_for_endpoints(endpoints, required_processes)
        for status in self._statuses:
            status.state = ReadinessState.READY
            status.endpoint_publication = EndpointPublicationState.PUBLISHED
            status.last_probe = _probe_evidence(self._now(), ProbeOutcome.SUCCESS, "endpoint_ready", "endpoint ready")
        self._publish_readiness(ReadinessState.READY)

        generation = self._client_steps.build_generation_step(
            self._context.plan,
            self._context.shard,
            self._attempt,
            self._context.attempt_directory,
            endpoints,
            self._environment,
        )
        generation_started_at = self._now()
        self._supervisor.wait(self._supervisor.start(generation), required=required_processes)
        self._supervisor.require_running(required_processes)
        client_result, candidate = load_complete_client_candidate(self._context)
        if candidate.created_at < generation_started_at or client_result.completed_at < generation_started_at:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.FINALIZATION_FAILED,
                "client result predates the current generation step",
            )
        if client_result.completed_at > self._now():
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.FINALIZATION_FAILED,
                "client completion timestamp is later than the allocation clock",
            )
        candidate_reference = client_result.candidate_output_manifest
        if candidate_reference is None:  # pragma: no cover - the record contract requires this for complete results
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.FINALIZATION_FAILED,
                "complete client result has no candidate reference",
            )
        return candidate_reference, client_result.completed_at

    def _validate_attempt_state(self) -> None:
        if self._attempt.state not in {
            AttemptLifecycleState.SUBMITTED,
            AttemptLifecycleState.PENDING,
        }:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.INVALID_CONTEXT,
                "allocation attempt is not executable",
            )

    def _mark_attempt_running(self) -> None:
        if self._attempt.state is AttemptLifecycleState.RUNNING:
            return
        self._attempt = _copy_attempt(
            self._attempt,
            state=AttemptLifecycleState.RUNNING,
            updated_at=self._now(),
        )
        self._attempt = self._state.update_attempt(self._attempt)

    def _start_servers(
        self,
        deployments: tuple[ResolvedVllmServerDeployment, ...],
    ) -> tuple[tuple[ManagedStep, ...], ...]:
        all_managed: list[tuple[ManagedStep, ...]] = []
        for deployment in deployments:
            started_at = self._clock.monotonic()
            managed: list[ManagedStep] = []
            steps = build_vllm_steps(
                deployment,
                self._context.plan,
                self._context.attempt_directory,
                self._environment,
            )
            for process, step in zip(deployment.processes, steps, strict=True):
                self._supervisor.require_running(tuple(managed))
                remaining_delay = process.launch_delay_seconds - (self._clock.monotonic() - started_at)
                if remaining_delay > 0:
                    self._clock.sleep(remaining_delay)
                managed.append(self._supervisor.start(step))
            all_managed.append(tuple(managed))
        return tuple(all_managed)

    def _wait_for_backends(self, status: _DeploymentStatus, required: tuple[ManagedStep, ...]) -> None:
        probes = status.deployment.readiness_probes
        deadline = self._clock.monotonic() + max(probe.deadline_seconds for probe in probes)
        previous_ready = -1
        while True:
            self._supervisor.require_running(required)
            ready = sum(
                self._prober.is_ready(
                    "127.0.0.1",
                    probe.port,
                    probe.path,
                    timeout_seconds=self._probe_timeout_seconds,
                )
                for probe in probes
            )
            if ready != previous_ready:
                status.ready_backends = ready
                status.last_probe = _probe_evidence(
                    self._now(),
                    ProbeOutcome.SUCCESS if ready == len(probes) else ProbeOutcome.FAILURE,
                    "backend_ready" if ready == len(probes) else "backend_starting",
                    f"{ready} of {len(probes)} backends ready",
                )
                self._publish_readiness(ReadinessState.STARTING)
                previous_ready = ready
            if ready == len(probes):
                return
            if self._clock.monotonic() >= deadline:
                raise SlurmRuntimeError(
                    SlurmRuntimeErrorCode.READINESS_TIMEOUT,
                    f"deployment {status.deployment.deployment_id!r} readiness timed out",
                )
            self._clock.sleep(self._poll_interval_seconds)

    def _wait_for_endpoints(
        self,
        endpoints: tuple[RuntimeEndpoint, ...],
        required: tuple[ManagedStep, ...],
    ) -> None:
        timeout = max(status.deployment.launch_policy.startup_timeout_seconds for status in self._statuses)
        deadline = self._clock.monotonic() + timeout
        while True:
            self._supervisor.require_running(required)
            if all(
                self._prober.is_ready(
                    endpoint.host,
                    endpoint.port,
                    "/health",
                    timeout_seconds=self._probe_timeout_seconds,
                )
                for endpoint in endpoints
            ):
                return
            if self._clock.monotonic() >= deadline:
                raise SlurmRuntimeError(
                    SlurmRuntimeErrorCode.READINESS_TIMEOUT,
                    "logical endpoint readiness timed out",
                )
            self._clock.sleep(self._poll_interval_seconds)

    def _publish_readiness(self, state: ReadinessState) -> None:
        revision = 1 if self._readiness is None else self._readiness.revision + 1
        readiness = AttemptReadiness(
            schema_version=1,
            run_id=self._attempt.run_id,
            shard_id=self._attempt.shard_id,
            attempt_id=self._attempt.attempt_id,
            revision=revision,
            updated_at=self._now(),
            state=state,
            deployments=tuple(
                DeploymentReadiness(
                    deployment_id=status.deployment.deployment_id,
                    model_alias=status.deployment.model_alias,
                    state=status.state,
                    expected_backends=len(status.deployment.backend_endpoints),
                    ready_backends=status.ready_backends,
                    endpoint_publication=status.endpoint_publication,
                    last_probe=status.last_probe,
                )
                for status in self._statuses
            ),
        )
        self._readiness = self._state.write_readiness(readiness)

    def _record_failed_readiness(self, failure: SlurmRuntimeError) -> SlurmRuntimeError:
        if self._readiness is None:
            return failure
        try:
            for status in self._statuses:
                status.state = ReadinessState.FAILED
                status.ready_backends = 0
                if status.endpoint_publication is EndpointPublicationState.PENDING:
                    status.endpoint_publication = EndpointPublicationState.FAILED
                status.last_probe = _probe_evidence(
                    self._now(),
                    ProbeOutcome.FAILURE,
                    failure.code.value,
                    "allocation runtime failed",
                )
            self._publish_readiness(ReadinessState.FAILED)
        except Exception as error:
            failure.add_note(f"failed readiness could not be persisted: {type(error).__name__}")
        return failure

    def _publish_stopped_readiness(self) -> None:
        for status in self._statuses:
            status.state = ReadinessState.STOPPED
            status.ready_backends = 0
            status.last_probe = _probe_evidence(
                self._now(),
                ProbeOutcome.SUCCESS,
                "runtime_stopped",
                "allocation processes stopped",
            )
        self._publish_readiness(ReadinessState.STOPPED)

    def _publish_terminal_attempt(
        self,
        *,
        failure: SlurmRuntimeError | None,
        candidate_reference: ArtifactReference | None,
        client_completed_at: datetime | None,
    ) -> AttemptManifest:
        timestamp = self._now()
        if client_completed_at is not None and client_completed_at > timestamp:
            timestamp = client_completed_at
        if failure is None:
            if candidate_reference is None:
                raise SlurmRuntimeError(
                    SlurmRuntimeErrorCode.FINALIZATION_FAILED,
                    "successful allocation has no candidate reference",
                )
            terminal = _copy_attempt(
                self._attempt,
                state=AttemptLifecycleState.SUCCEEDED,
                terminal_classification=AttemptTerminalClassification.SUCCEEDED,
                candidate_output=candidate_reference,
                updated_at=timestamp,
            )
        else:
            terminal = _copy_attempt(
                self._attempt,
                state=AttemptLifecycleState.FAILED,
                terminal_classification=AttemptTerminalClassification.FAILED,
                candidate_output=None,
                updated_at=timestamp,
            )
        self._attempt = self._state.update_attempt(terminal)
        return self._attempt

    def _now(self) -> datetime:
        value = self._clock.now()
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime clock is not UTC")
        minimum = self._attempt.updated_at
        if self._readiness is not None and self._readiness.updated_at > minimum:
            minimum = self._readiness.updated_at
        if value < minimum:
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime clock moved backward")
        return value


def _copy_attempt(attempt: AttemptManifest, **updates: object) -> AttemptManifest:
    payload = attempt.model_dump(mode="python")
    payload.update(updates)
    return AttemptManifest.model_validate(payload)


def _probe_evidence(
    observed_at: datetime,
    outcome: ProbeOutcome,
    reason_code: str,
    message: str,
) -> ProbeEvidence:
    return ProbeEvidence(
        observed_at=observed_at,
        outcome=outcome,
        reason_code=reason_code,
        redacted_message=message,
    )


def _normalize_failure(error: BaseException) -> SlurmRuntimeError:
    if isinstance(error, SlurmRuntimeError):
        return error
    if isinstance(error, (OSError, subprocess.SubprocessError)):
        return SlurmRuntimeError(SlurmRuntimeErrorCode.STEP_FAILED, "allocation process operation failed")
    if isinstance(error, KeyboardInterrupt):
        return SlurmRuntimeError(SlurmRuntimeErrorCode.CLIENT_FAILED, "allocation execution was interrupted")
    return SlurmRuntimeError(
        SlurmRuntimeErrorCode.FINALIZATION_FAILED,
        "allocation runtime failed at an internal boundary",
    )


__all__ = ["OneNodeAllocationController", "RuntimeStateStore"]
