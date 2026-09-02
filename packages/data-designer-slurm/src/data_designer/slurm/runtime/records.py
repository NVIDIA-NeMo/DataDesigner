# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and validate #876-owned client and candidate records."""

from __future__ import annotations

import posixpath
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from data_designer.slurm.client import ClientOutcome, ClientResult
from data_designer.slurm.contracts import ContractRecord
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.runtime.models import AllocationContext
from data_designer.slurm.state import CandidateOutputManifest
from data_designer.slurm.state.filesystem import open_verified_directory, read_regular_text

_CLIENT_RESULT_NAME = "client-result.json"
_CANDIDATE_NAME = "output-manifest.json"
_MAXIMUM_RECORD_SIZE = 16 * 1024 * 1024
_RecordT = TypeVar("_RecordT", bound=ContractRecord)


def load_complete_client_candidate(
    context: AllocationContext,
) -> tuple[ClientResult, CandidateOutputManifest]:
    """Load a complete semantic client result and its digest-bound candidate."""
    client_result = _read_record(context.attempt_directory, _CLIENT_RESULT_NAME, ClientResult)
    if client_result.outcome is not ClientOutcome.COMPLETE:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.CLIENT_FAILED,
            f"client generation finished with outcome {client_result.outcome.value!r}",
        )
    reference = client_result.candidate_output_manifest
    if reference is None:  # pragma: no cover - ClientResult validates this invariant
        raise SlurmRuntimeError(SlurmRuntimeErrorCode.FINALIZATION_FAILED, "client result has no candidate output")
    expected_path = context.attempt_directory / _CANDIDATE_NAME
    if reference.path != expected_path.as_posix():
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.FINALIZATION_FAILED,
            "client candidate path does not match the allocation attempt",
        )
    candidate = _read_record(context.attempt_directory, _CANDIDATE_NAME, CandidateOutputManifest)
    if candidate.compute_sha256() != reference.sha256:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.FINALIZATION_FAILED,
            "client candidate digest does not match the persisted manifest",
        )
    _validate_candidate(context, client_result, candidate)
    return client_result, candidate


def _validate_candidate(
    context: AllocationContext,
    client_result: ClientResult,
    candidate: CandidateOutputManifest,
) -> None:
    _validate_candidate_identities(context, client_result, candidate)
    _validate_candidate_counts(context, client_result, candidate)
    _validate_candidate_location(context, client_result, candidate)
    if candidate.created_at < context.attempt.created_at or client_result.completed_at < candidate.created_at:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.FINALIZATION_FAILED,
            "client and candidate timestamps are inconsistent",
        )


def _validate_candidate_identities(
    context: AllocationContext,
    client_result: ClientResult,
    candidate: CandidateOutputManifest,
) -> None:
    identities = (
        client_result.run_id == candidate.run_id == context.plan.run_id,
        client_result.shard_id == candidate.shard_id == context.shard.shard_id,
        client_result.attempt_id == candidate.attempt_id == context.attempt.attempt_id,
        candidate.attempt_ordinal == context.attempt.attempt_ordinal,
    )
    if not all(identities):
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.FINALIZATION_FAILED,
            "client and candidate identities do not match the allocation",
        )


def _validate_candidate_counts(
    context: AllocationContext,
    client_result: ClientResult,
    candidate: CandidateOutputManifest,
) -> None:
    if not candidate.winner_eligible:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.FINALIZATION_FAILED,
            "candidate record counts do not complete the planned shard",
        )
    if (
        client_result.requested_records != context.shard.requested_records
        or client_result.actual_records != candidate.actual_records
        or candidate.actual_records != context.shard.requested_records
    ):
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.FINALIZATION_FAILED,
            "client and candidate record counts do not complete the planned shard",
        )


def _validate_candidate_location(
    context: AllocationContext,
    client_result: ClientResult,
    candidate: CandidateOutputManifest,
) -> None:
    if client_result.requested_resume_mode != context.plan.invocation.authored.resume:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.FINALIZATION_FAILED,
            "client resume mode does not match the resolved plan",
        )
    expected_dataset_path = (
        context.shard.resume_workspace.path
        if client_result.effective_resume_mode == "always"
        else posixpath.join(context.attempt_directory.as_posix(), "dataset")
    )
    if client_result.dataset_path != expected_dataset_path or candidate.dataset_path != expected_dataset_path:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.FINALIZATION_FAILED,
            "client and candidate dataset paths do not match the resolved plan",
        )


def _read_record(directory: Path, name: str, record_type: type[_RecordT]) -> _RecordT:
    display_path = directory / name
    try:
        with open_verified_directory(directory, require_private=True) as descriptor:
            content = read_regular_text(
                descriptor,
                name,
                display_path,
                maximum_size=_MAXIMUM_RECORD_SIZE,
            )
        return record_type.model_validate_json(content)
    except (OSError, UnicodeError, ValidationError, ValueError) as error:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.FINALIZATION_FAILED,
            f"runtime result record {name!r} is unavailable or invalid",
        ) from error


__all__ = ["load_complete_client_candidate"]
