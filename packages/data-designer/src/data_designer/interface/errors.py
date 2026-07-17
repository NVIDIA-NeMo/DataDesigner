# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.errors import DataDesignerError


class DataDesignerProfilingError(DataDesignerError):
    """Raised for errors related to a Data Designer dataset profiling."""


class DataDesignerGenerationError(DataDesignerError):
    """Raised for errors related to a Data Designer dataset generation."""


class DataDesignerWorkflowError(DataDesignerError):
    """Raised for errors related to composite workflow orchestration."""


class RecordRetryExhaustedError(DataDesignerWorkflowError):
    """Raised when a record retry reaches its bounds before every slot passes."""

    target_records: int
    accepted_records: int
    unresolved_records: int
    candidate_records: int
    attempts: int
    unresolved_slot_ids: tuple[int, ...]

    def __init__(
        self,
        *,
        target_records: int,
        accepted_records: int,
        candidate_records: int,
        attempts: int,
        unresolved_slot_ids: list[int],
    ) -> None:
        self.target_records = target_records
        self.accepted_records = accepted_records
        self.unresolved_records = len(unresolved_slot_ids)
        self.candidate_records = candidate_records
        self.attempts = attempts
        self.unresolved_slot_ids = tuple(unresolved_slot_ids)
        super().__init__(
            f"Record retry exhausted after {attempts} attempt(s): accepted {accepted_records} of "
            f"{target_records} slots after {candidate_records} candidate records."
        )


class DataDesignerEarlyShutdownError(DataDesignerGenerationError):
    """Raised when a run terminated via early shutdown and produced no records.

    Subclass of ``DataDesignerGenerationError`` so existing handlers still catch
    it; callers that want to distinguish the early-shutdown case (e.g. to retry
    with a different model alias or surface a degraded-provider message to the
    user) can catch this specific type.
    """


class InvalidBufferValueError(DataDesignerError):
    """Raised for errors related to an invalid buffer value."""
