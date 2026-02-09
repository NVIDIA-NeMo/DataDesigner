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


class ToolUsageStats(BaseModel):
    total_tool_calls: int = 0
    total_tool_call_turns: int = 0
    total_generations: int = 0
    generations_with_tools: int = 0

    @property
    def has_usage(self) -> bool:
        return self.total_generations > 0

    def extend(self, *, tool_calls: int, tool_call_turns: int) -> None:
        """Extend stats with a single generation's tool usage."""
        self.total_generations += 1
        self.total_tool_calls += tool_calls
        self.total_tool_call_turns += tool_call_turns
        if tool_calls > 0:
            self.generations_with_tools += 1

    def merge(self, other: ToolUsageStats) -> ToolUsageStats:
        """Merge another ToolUsageStats object."""
        self.total_tool_calls += other.total_tool_calls
        self.total_tool_call_turns += other.total_tool_call_turns
        self.total_generations += other.total_generations
        self.generations_with_tools += other.generations_with_tools
        return self


class ImageUsageStats(BaseModel):
    total_images: int = 0

    @property
    def has_usage(self) -> bool:
        return self.total_images > 0

    def extend(self, *, images: int) -> None:
        """Extend stats with generated images count."""
        self.total_images += images


class ModelUsageStats(BaseModel):
    token_usage: TokenUsageStats = TokenUsageStats()
    request_usage: RequestUsageStats = RequestUsageStats()
    tool_usage: ToolUsageStats = ToolUsageStats()
    image_usage: ImageUsageStats = ImageUsageStats()

    @property
    def has_usage(self) -> bool:
        return self.token_usage.has_usage and self.request_usage.has_usage

    def extend(
        self,
        *,
        token_usage: TokenUsageStats | None = None,
        request_usage: RequestUsageStats | None = None,
        tool_usage: ToolUsageStats | None = None,
        image_usage: ImageUsageStats | None = None,
    ) -> None:
        if token_usage is not None:
            self.token_usage.extend(input_tokens=token_usage.input_tokens, output_tokens=token_usage.output_tokens)
        if request_usage is not None:
            self.request_usage.extend(
                successful_requests=request_usage.successful_requests, failed_requests=request_usage.failed_requests
            )
        if tool_usage is not None:
            self.tool_usage.merge(tool_usage)
        if image_usage is not None:
            self.image_usage.extend(images=image_usage.total_images)

    def get_usage_stats(self, *, total_time_elapsed: float) -> dict:
        exclude = set()
        if not self.tool_usage.has_usage:
            exclude.add("tool_usage")
        if not self.image_usage.has_usage:
            exclude.add("image_usage")
        exclude = exclude if exclude else None
        return self.model_dump(exclude=exclude) | {
            "tokens_per_second": int(self.token_usage.total_tokens / total_time_elapsed)
            if total_time_elapsed > 0
            else 0,
            "requests_per_minute": int(self.request_usage.total_requests / total_time_elapsed * 60)
            if total_time_elapsed > 0
            else 0,
        }
