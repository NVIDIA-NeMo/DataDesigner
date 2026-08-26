# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable errors raised by public Slurm services."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TypeVar

_ResultT = TypeVar("_ResultT")


class SlurmServiceErrorCode(str, Enum):
    """Machine-readable public service error categories."""

    INVALID_REQUEST = "invalid_request"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    UNAVAILABLE = "unavailable"
    INTERNAL = "internal"


class SlurmServiceOperation(str, Enum):
    """Public operations with stable error attribution."""

    PLAN_RUN = "plan_run"
    RESOLVE_IMAGE = "resolve_image"
    RUN_BENCHMARK = "run_benchmark"
    ANALYZE_BENCHMARK = "analyze_benchmark"


class SlurmServiceError(RuntimeError):
    """Normalized public service failure with a stable code and operation."""

    def __init__(
        self,
        code: SlurmServiceErrorCode,
        operation: SlurmServiceOperation,
        message: str,
    ) -> None:
        if not isinstance(code, SlurmServiceErrorCode):
            raise TypeError("code must be a SlurmServiceErrorCode")
        if not isinstance(operation, SlurmServiceOperation):
            raise TypeError("operation must be a SlurmServiceOperation")
        if type(message) is not str or not message or len(message) > 512:
            raise ValueError("service error message must contain 1 to 512 characters")
        if any(ord(character) < 32 or ord(character) == 127 for character in message):
            raise ValueError("service error message must not contain control characters")
        self.code = code
        self.operation = operation
        super().__init__(message)


def invalid_request(operation: SlurmServiceOperation, message: str) -> SlurmServiceError:
    """Build one normalized invalid-request error."""
    return SlurmServiceError(SlurmServiceErrorCode.INVALID_REQUEST, operation, message)


def invoke_backend(operation: SlurmServiceOperation, call: Callable[[], _ResultT]) -> _ResultT:
    """Preserve normalized failures and redact unexpected backend exceptions."""
    try:
        return call()
    except SlurmServiceError as error:
        if error.operation is operation:
            raise
    except Exception:
        pass
    raise SlurmServiceError(
        SlurmServiceErrorCode.INTERNAL,
        operation,
        f"{operation.value.replace('_', ' ')} failed",
    ) from None
