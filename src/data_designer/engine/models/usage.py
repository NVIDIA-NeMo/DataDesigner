# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from pydantic import BaseModel, computed_field

logger = logging.getLogger(__name__)


class TokenUsageStats(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @computed_field
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def has_usage(self) -> bool:
        return self.total_tokens > 0

    def extend(self, *, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens


class RequestUsageStats(BaseModel):
    successful_requests: int = 0
    failed_requests: int = 0

    @computed_field
    def total_requests(self) -> int:
        return self.successful_requests + self.failed_requests

    @property
    def has_usage(self) -> bool:
        return self.total_requests > 0

    def extend(self, *, successful_requests: int, failed_requests: int) -> None:
        self.successful_requests += successful_requests
        self.failed_requests += failed_requests


class ModelUsageStats(BaseModel):
    token_usage: TokenUsageStats = TokenUsageStats()
    request_usage: RequestUsageStats = RequestUsageStats()

    @property
    def has_usage(self) -> bool:
        return self.token_usage.has_usage and self.request_usage.has_usage

    def extend(
        self, *, token_usage: TokenUsageStats | None = None, request_usage: RequestUsageStats | None = None
    ) -> None:
        if token_usage is not None:
            self.token_usage.extend(input_tokens=token_usage.input_tokens, output_tokens=token_usage.output_tokens)
        if request_usage is not None:
            self.request_usage.extend(
                successful_requests=request_usage.successful_requests, failed_requests=request_usage.failed_requests
            )

    def get_usage_stats(self, *, total_time_elapsed: float) -> dict:
        return self.model_dump() | {
            "tokens_per_second": int(self.token_usage.total_tokens / total_time_elapsed)
            if total_time_elapsed > 0
            else 0,
            "requests_per_minute": int(self.request_usage.total_requests / total_time_elapsed * 60)
            if total_time_elapsed > 0
            else 0,
        }
