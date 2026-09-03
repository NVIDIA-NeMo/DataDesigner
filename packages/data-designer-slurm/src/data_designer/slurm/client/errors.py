# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.slurm.client.records import ClientErrorCode


class ClientWorkerError(RuntimeError):
    """Failure safe to classify at the allocation boundary."""

    def __init__(self, code: ClientErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.redacted_message = message
