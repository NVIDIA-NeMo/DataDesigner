# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and validate #876-owned client and candidate records."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from data_designer.slurm.client import ClientOutcome, ClientResult
from data_designer.slurm.contracts import ContractRecord
from data_designer.slurm.integration import IntegrationContractError, PlanStateValidator
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
    candidate = _read_record(context.attempt_directory, _CANDIDATE_NAME, CandidateOutputManifest)
    try:
        PlanStateValidator(context.plan).validate_client_candidate(
            context.shard,
            context.attempt,
            client_result,
            candidate,
        )
    except IntegrationContractError as error:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.FINALIZATION_FAILED,
            "client result and candidate do not match the resolved plan",
        ) from error
    return client_result, candidate


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
