# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

from data_designer.config.terminal_failure import TerminalTaskFailure
from data_designer.errors import DataDesignerError


class DataDesignerProfilingError(DataDesignerError):
    """Raised for errors related to a Data Designer dataset profiling."""


class DataDesignerGenerationError(DataDesignerError):
    """Raised for errors related to a Data Designer dataset generation."""

    def __init__(
        self,
        *args: object,
        terminal_failures: Sequence[TerminalTaskFailure] | None = None,
    ) -> None:
        super().__init__(*args)
        self.terminal_failures = list(terminal_failures or [])


class DataDesignerWorkflowError(DataDesignerError):
    """Raised for errors related to composite workflow orchestration."""


class DataDesignerEarlyShutdownError(DataDesignerGenerationError):
    """Raised when a run terminated via early shutdown and produced no records.

    Subclass of ``DataDesignerGenerationError`` so existing handlers still catch
    it; callers that want to distinguish the early-shutdown case (e.g. to retry
    with a different model alias or surface a degraded-provider message to the
    user) can catch this specific type.
    """


class InvalidBufferValueError(DataDesignerError):
    """Raised for errors related to an invalid buffer value."""
