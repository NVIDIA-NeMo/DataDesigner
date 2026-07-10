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


class DataDesignerEarlyShutdownError(DataDesignerGenerationError):
    """Raised when a run terminated via early shutdown and produced no records.

    Subclass of ``DataDesignerGenerationError`` so existing handlers still catch
    it; callers that want to distinguish the early-shutdown case (e.g. to retry
    with a different model alias or surface a degraded-provider message to the
    user) can catch this specific type.
    """


class DataDesignerRecordSelectionExhaustedError(DataDesignerGenerationError):
    """Raised when record selection exhausts its configured candidate budget."""

    def __init__(
        self,
        *,
        target_records: int,
        accepted_records: int,
        candidate_records: int,
        max_candidate_records: int,
    ) -> None:
        self.target_records = target_records
        self.accepted_records = accepted_records
        self.candidate_records = candidate_records
        self.max_candidate_records = max_candidate_records
        super().__init__(
            "🛑 Record selection exhausted its candidate budget: "
            f"accepted {accepted_records} of {target_records} requested records after generating "
            f"{candidate_records} of {max_candidate_records} allowed candidates."
        )


class InvalidBufferValueError(DataDesignerError):
    """Raised for errors related to an invalid buffer value."""
