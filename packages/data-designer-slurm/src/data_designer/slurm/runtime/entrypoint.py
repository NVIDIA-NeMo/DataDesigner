# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed control phases executed only inside the sealed client image."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data_designer.slurm.client.filesystem import ensure_private_directory, replace_private_text
from data_designer.slurm.runtime.bootstrap import build_runtime_manifest
from data_designer.slurm.runtime.context import load_allocation_context
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.runtime.logs import execution_log_directory
from data_designer.slurm.runtime.models import AllocationContext
from data_designer.slurm.runtime.paths import get_container_path
from data_designer.slurm.runtime.ports import resolve_allocation_deployments
from data_designer.slurm.runtime.preflight import SystemAllocationPreflight
from data_designer.slurm.runtime.records import load_complete_client_candidate
from data_designer.slurm.serving.deployment import ResolvedVllmServerDeployment
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
    SlurmStateWriter,
    StateNotFoundError,
)


def main(arguments: Sequence[str] | None = None) -> int:
    """Execute one container-only allocation control phase."""
    parsed = _parse_arguments(arguments)
    try:
        if parsed.operation == "prepare":
            _prepare(parsed, os.environ)
        elif parsed.operation == "ready":
            _ready(parsed, os.environ)
        elif parsed.operation == "client":
            _client(parsed, os.environ)
        elif parsed.operation == "succeed":
            _succeed(parsed, os.environ)
        else:
            _fail(parsed, os.environ)
    except SlurmRuntimeError as error:
        print(f"allocation runtime failed ({error.code.value}): {error}", file=sys.stderr)
        return (
            64 if error.code in {SlurmRuntimeErrorCode.INVALID_CONTEXT, SlurmRuntimeErrorCode.PREFLIGHT_FAILED} else 70
        )
    except KeyboardInterrupt:
        print("allocation runtime interrupted", file=sys.stderr)
        return 130
    except Exception:
        print("allocation runtime failed at an internal boundary", file=sys.stderr)
        return 70
    return 0


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="data-designer-slurm-runtime")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("prepare", "ready", "succeed", "fail"):
        subparser = subparsers.add_parser(operation)
        _add_context_arguments(subparser)
    prepare = subparsers.choices["prepare"]
    prepare.add_argument("--runtime-root", required=True, type=Path)
    prepare.add_argument("--manifest", required=True, type=Path)
    client = subparsers.add_parser("client")
    _add_context_arguments(client)
    client.add_argument("--endpoint", action="append", default=[])
    return parser.parse_args(arguments)


def _add_context_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--retry-id")
    parser.add_argument("--retry-plan-sha256")
    parser.add_argument("--effective-resume-mode", choices=("never", "always"))


def _prepare(arguments: argparse.Namespace, environment: Mapping[str, str]) -> None:
    context, writer = _load_context(arguments, environment)
    _validate_attempt_is_executable(context.attempt)
    SystemAllocationPreflight.verify_attempt_directory(arguments.attempt_dir)
    SystemAllocationPreflight.verify_ports(context, environment)
    readiness = _begin_attempt(context, writer, environment)
    log_directory = execution_log_directory(context.attempt_directory, readiness.revision)
    container_log_directory = Path(get_container_path(context.plan, log_directory.as_posix(), require_writable=True))
    ensure_private_directory(container_log_directory)
    manifest = build_runtime_manifest(
        context,
        environment,
        runtime_root=arguments.runtime_root,
        log_directory=log_directory,
    )
    expected_manifest = context.attempt_directory / "runtime-manifest.json"
    if arguments.manifest.as_posix() != get_container_path(
        context.plan,
        expected_manifest.as_posix(),
        require_writable=True,
    ):
        raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime manifest path is invalid")
    replace_private_text(arguments.manifest, manifest.serialize_json())


def _ready(arguments: argparse.Namespace, environment: Mapping[str, str]) -> None:
    context, writer = _load_context(arguments, environment)
    previous = writer.load_readiness(context.shard.shard_id, context.attempt.attempt_id)
    timestamp = _now(context.attempt, previous)
    deployments = _resolve_deployments(context, environment)
    writer.write_readiness(
        AttemptReadiness(
            schema_version=1,
            run_id=context.plan.run_id,
            shard_id=context.shard.shard_id,
            attempt_id=context.attempt.attempt_id,
            revision=previous.revision + 1,
            updated_at=timestamp,
            state=ReadinessState.READY,
            deployments=tuple(
                DeploymentReadiness(
                    deployment_id=deployment.deployment_id,
                    model_alias=deployment.model_alias,
                    state=ReadinessState.READY,
                    expected_backends=len(deployment.backend_endpoints),
                    ready_backends=len(deployment.backend_endpoints),
                    endpoint_publication=EndpointPublicationState.PUBLISHED,
                    last_probe=_probe(timestamp, ProbeOutcome.SUCCESS, "endpoint_ready", "endpoint ready"),
                )
                for deployment in deployments
            ),
        )
    )


def _client(arguments: argparse.Namespace, environment: Mapping[str, str]) -> None:
    context, writer = _load_context(arguments, environment)
    generation_started_at = _now(context.attempt, _load_optional_readiness(context, writer))
    resume_mode = (
        context.plan.invocation.authored.resume
        if context.retry_plan is None
        else context.retry_plan.effective_resume_mode
    )
    with writer.acquire_dataset_workspace(
        context.shard.shard_id,
        context.attempt.attempt_id,
        resume_mode,
    ):
        return_code = _run_client_worker(
            (
                "run",
                "--plan",
                arguments.plan.as_posix(),
                "--shard-id",
                context.shard.shard_id,
                "--attempt-id",
                context.attempt.attempt_id,
                "--attempt-dir",
                arguments.attempt_dir.as_posix(),
                *(() if context.retry_plan is None else ("--resume-mode", context.retry_plan.effective_resume_mode)),
                *(argument for endpoint in arguments.endpoint for argument in ("--endpoint", endpoint)),
            )
        )
        if return_code != 0:
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.CLIENT_FAILED, "client generation failed")
        client_result, candidate = load_complete_client_candidate(
            context,
            context.attempt,
            attempt_directory=arguments.attempt_dir,
        )
        if (
            context.retry_plan is not None
            and client_result.effective_resume_mode != context.retry_plan.effective_resume_mode
        ):
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.FINALIZATION_FAILED,
                "client effective resume mode differs from the persisted retry plan",
            )
        completed_at = client_result.completed_at
        if candidate.created_at < generation_started_at or completed_at < generation_started_at:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.FINALIZATION_FAILED,
                "client result predates the current generation step",
            )
        if completed_at > datetime.now(timezone.utc):
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.FINALIZATION_FAILED,
                "client completion timestamp is later than the allocation clock",
            )
        writer.publish_attempt_result(client_result, candidate)


def _run_client_worker(arguments: Sequence[str]) -> int:
    command = (sys.executable, "-m", "data_designer.slurm.client.worker", *arguments)
    return subprocess.run(command, check=False).returncode


def _succeed(arguments: argparse.Namespace, environment: Mapping[str, str]) -> None:
    context, writer = _load_context(arguments, environment)
    stopped_at = _write_stopped_readiness(context, writer)
    attempt = writer.load_attempt(context.shard.shard_id, context.attempt.attempt_id)
    if attempt.candidate_output is None:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.FINALIZATION_FAILED,
            "successful allocation has no candidate reference",
        )
    completed_at = max(stopped_at, attempt.updated_at)
    writer.finalize_winner(
        attempt.shard_id,
        attempt.attempt_id,
        completed_at=completed_at,
        published_at=max(datetime.now(timezone.utc), completed_at),
    )


def _fail(arguments: argparse.Namespace, environment: Mapping[str, str]) -> None:
    context, writer = _load_context(arguments, environment)
    attempt = writer.load_attempt(context.shard.shard_id, context.attempt.attempt_id)
    if attempt.state in {AttemptLifecycleState.SUCCEEDED, AttemptLifecycleState.FAILED}:
        return
    timestamp = _write_failed_and_stopped_readiness(context, writer)
    writer.update_attempt(
        attempt.model_copy(
            update={
                "state": AttemptLifecycleState.FAILED,
                "terminal_classification": AttemptTerminalClassification.FAILED,
                "updated_at": max(timestamp, attempt.updated_at),
            }
        )
    )


def _load_context(
    arguments: argparse.Namespace,
    environment: Mapping[str, str],
) -> tuple[AllocationContext, SlurmStateWriter]:
    return load_allocation_context(
        arguments.plan,
        arguments.attempt_dir,
        environment,
        retry_id=arguments.retry_id,
        retry_plan_sha256=arguments.retry_plan_sha256,
        effective_resume_mode=arguments.effective_resume_mode,
    )


def _begin_attempt(
    context: AllocationContext,
    writer: SlurmStateWriter,
    environment: Mapping[str, str],
) -> AttemptReadiness:
    attempt = context.attempt
    previous = _load_optional_readiness(context, writer)
    timestamp = _now(attempt, previous)
    if attempt.state is not AttemptLifecycleState.RUNNING:
        attempt = writer.update_attempt(
            attempt.model_copy(update={"state": AttemptLifecycleState.RUNNING, "updated_at": timestamp})
        )
    deployments = _resolve_deployments(context, environment)
    initial_state = ReadinessState.RESTARTING if previous is not None else ReadinessState.PENDING
    initial = writer.write_readiness(_readiness(context, deployments, previous, initial_state, timestamp))
    return writer.write_readiness(
        _readiness(context, deployments, initial, ReadinessState.STARTING, _now(attempt, initial))
    )


def _readiness(
    context: AllocationContext,
    deployments: tuple[ResolvedVllmServerDeployment, ...],
    previous: AttemptReadiness | None,
    state: ReadinessState,
    timestamp: datetime,
) -> AttemptReadiness:
    return AttemptReadiness(
        schema_version=1,
        run_id=context.plan.run_id,
        shard_id=context.shard.shard_id,
        attempt_id=context.attempt.attempt_id,
        revision=1 if previous is None else previous.revision + 1,
        updated_at=timestamp,
        state=state,
        deployments=tuple(
            DeploymentReadiness(
                deployment_id=deployment.deployment_id,
                model_alias=deployment.model_alias,
                state=state,
                expected_backends=len(deployment.backend_endpoints),
                ready_backends=0,
                endpoint_publication=EndpointPublicationState.PENDING,
            )
            for deployment in deployments
        ),
    )


def _write_failed_and_stopped_readiness(context: AllocationContext, writer: SlurmStateWriter) -> datetime:
    previous = _load_optional_readiness(context, writer)
    if previous is None:
        return max(datetime.now(timezone.utc), context.attempt.updated_at)
    failed_at = _now(context.attempt, previous)
    failed = writer.write_readiness(
        AttemptReadiness(
            schema_version=1,
            run_id=previous.run_id,
            shard_id=previous.shard_id,
            attempt_id=previous.attempt_id,
            revision=previous.revision + 1,
            updated_at=failed_at,
            state=ReadinessState.FAILED,
            deployments=tuple(
                deployment.model_copy(
                    update={
                        "state": ReadinessState.FAILED,
                        "ready_backends": 0,
                        "endpoint_publication": (
                            EndpointPublicationState.FAILED
                            if deployment.endpoint_publication is EndpointPublicationState.PENDING
                            else deployment.endpoint_publication
                        ),
                        "last_probe": _probe(
                            failed_at,
                            ProbeOutcome.FAILURE,
                            "allocation_failed",
                            "allocation runtime failed",
                        ),
                    }
                )
                for deployment in previous.deployments
            ),
        )
    )
    return _write_stopped_readiness(context, writer, previous=failed)


def _write_stopped_readiness(
    context: AllocationContext,
    writer: SlurmStateWriter,
    *,
    previous: AttemptReadiness | None = None,
) -> datetime:
    previous = previous or writer.load_readiness(context.shard.shard_id, context.attempt.attempt_id)
    timestamp = _now(context.attempt, previous)
    writer.write_readiness(
        AttemptReadiness(
            schema_version=1,
            run_id=previous.run_id,
            shard_id=previous.shard_id,
            attempt_id=previous.attempt_id,
            revision=previous.revision + 1,
            updated_at=timestamp,
            state=ReadinessState.STOPPED,
            deployments=tuple(
                deployment.model_copy(
                    update={
                        "state": ReadinessState.STOPPED,
                        "ready_backends": 0,
                        "last_probe": _probe(
                            timestamp,
                            ProbeOutcome.SUCCESS,
                            "runtime_stopped",
                            "allocation processes stopped",
                        ),
                    }
                )
                for deployment in previous.deployments
            ),
        )
    )
    return timestamp


def _load_optional_readiness(context: AllocationContext, writer: SlurmStateWriter) -> AttemptReadiness | None:
    try:
        return writer.load_readiness(context.shard.shard_id, context.attempt.attempt_id)
    except StateNotFoundError:
        return None


def _resolve_deployments(
    context: AllocationContext,
    environment: Mapping[str, str],
) -> tuple[ResolvedVllmServerDeployment, ...]:
    return resolve_allocation_deployments(context, environment)


def _validate_attempt_is_executable(attempt: AttemptManifest) -> None:
    if attempt.state not in {
        AttemptLifecycleState.SUBMITTED,
        AttemptLifecycleState.PENDING,
        AttemptLifecycleState.RUNNING,
    }:
        raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "allocation attempt is not executable")


def _probe(observed_at: datetime, outcome: ProbeOutcome, reason_code: str, message: str) -> ProbeEvidence:
    return ProbeEvidence(
        observed_at=observed_at,
        outcome=outcome,
        reason_code=reason_code,
        redacted_message=message,
    )


def _now(attempt: AttemptManifest, readiness: AttemptReadiness | None) -> datetime:
    value = datetime.now(timezone.utc)
    minimum = attempt.updated_at
    if readiness is not None and readiness.updated_at > minimum:
        minimum = readiness.updated_at
    if value < minimum:
        raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime clock moved backward")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):  # pragma: no cover - system clock is UTC
        raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime clock is not UTC")
    return value


if __name__ == "__main__":  # pragma: no cover - exercised through the installed module
    raise SystemExit(main())
