# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.engine.errors import DataDesignerError


class ArtifactStorageError(DataDesignerError): ...


class DatasetGenerationError(DataDesignerError): ...


class RecordSelectionEarlyShutdownError(DatasetGenerationError):
    """Raised when record selection stops at the scheduler's early-shutdown gate."""

    def __init__(self, *, candidate_budget_remaining: bool = True) -> None:
        message = "🛑 Record selection stopped after the scheduler triggered early shutdown."
        if candidate_budget_remaining:
            message += " Committed accepted batches can be continued with resume=ResumeMode.ALWAYS."
        super().__init__(message)


class RecordSelectionExhaustedError(DatasetGenerationError):
    """Raised when record selection consumes its candidate budget before reaching the target."""

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


class DatasetProcessingError(DataDesignerError): ...
