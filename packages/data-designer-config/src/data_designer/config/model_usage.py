# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelUsageSummary(BaseModel):
    """Usage recorded for one model alias during a single run.

    ``cost`` and ``currency`` are reserved and currently always ``None``.
    Counts are per logical request, not per provider retry. Token counts read 0
    when the provider omits usage.
    """

    model_config = ConfigDict(frozen=True)

    model_alias: str
    model_name: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None = None
    successful_requests: int
    failed_requests: int
    cost: float | None = None
    currency: str | None = None
