# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalized failures for allocation-local Slurm execution."""

from __future__ import annotations

from enum import Enum


class SlurmRuntimeErrorCode(str, Enum):
    """Stable allocation-runtime failure categories."""

    INVALID_CONTEXT = "invalid_context"
    PREFLIGHT_FAILED = "preflight_failed"
    STEP_FAILED = "step_failed"
    READINESS_TIMEOUT = "readiness_timeout"
    CLIENT_FAILED = "client_failed"
    FINALIZATION_FAILED = "finalization_failed"
    CLEANUP_FAILED = "cleanup_failed"


class SlurmRuntimeError(RuntimeError):
    """Bounded allocation-runtime error safe for persisted diagnostics."""

    def __init__(self, code: SlurmRuntimeErrorCode, message: str) -> None:
        if not isinstance(code, SlurmRuntimeErrorCode):
            raise TypeError("code must be a SlurmRuntimeErrorCode")
        if type(message) is not str or not message or len(message) > 512:
            raise ValueError("runtime error message must contain 1 to 512 characters")
        if any(ord(character) < 32 or ord(character) == 127 for character in message):
            raise ValueError("runtime error message must not contain control characters")
        self.code = code
        super().__init__(message)
