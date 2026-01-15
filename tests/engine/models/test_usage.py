# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.engine.models.usage import ModelUsageStats, RequestUsageStats, TokenUsageStats


def test_token_usage_stats():
    token_usage_stats = TokenUsageStats()
    assert token_usage_stats.input_tokens == 0
    assert token_usage_stats.output_tokens == 0
    assert token_usage_stats.total_tokens == 0
    assert token_usage_stats.has_usage is False

    token_usage_stats.extend(input_tokens=10, output_tokens=20)
    assert token_usage_stats.input_tokens == 10
    assert token_usage_stats.output_tokens == 20
    assert token_usage_stats.total_tokens == 30
    assert token_usage_stats.has_usage is True


def test_request_usage_stats():
    request_usage_stats = RequestUsageStats()
    assert request_usage_stats.successful_requests == 0
    assert request_usage_stats.failed_requests == 0
    assert request_usage_stats.total_requests == 0
    assert request_usage_stats.has_usage is False

    request_usage_stats.extend(successful_requests=10, failed_requests=20)
    assert request_usage_stats.successful_requests == 10
    assert request_usage_stats.failed_requests == 20
    assert request_usage_stats.total_requests == 30
    assert request_usage_stats.has_usage is True


def test_model_usage_stats():
    model_usage_stats = ModelUsageStats()
    assert model_usage_stats.token_usage.input_tokens == 0
    assert model_usage_stats.token_usage.output_tokens == 0
    assert model_usage_stats.request_usage.successful_requests == 0
    assert model_usage_stats.request_usage.failed_requests == 0
    assert model_usage_stats.has_usage is False

    assert model_usage_stats.get_usage_stats(total_time_elapsed=10) == {
        "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "request_usage": {"successful_requests": 0, "failed_requests": 0, "total_requests": 0},
        "tokens_per_second": 0,
        "requests_per_minute": 0,
    }

    model_usage_stats.extend(
        token_usage=TokenUsageStats(input_tokens=10, output_tokens=20),
        request_usage=RequestUsageStats(successful_requests=2, failed_requests=1),
    )
    assert model_usage_stats.token_usage.input_tokens == 10
    assert model_usage_stats.token_usage.output_tokens == 20
    assert model_usage_stats.request_usage.successful_requests == 2
    assert model_usage_stats.request_usage.failed_requests == 1
    assert model_usage_stats.has_usage is True

    assert model_usage_stats.get_usage_stats(total_time_elapsed=2) == {
        "token_usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        "request_usage": {"successful_requests": 2, "failed_requests": 1, "total_requests": 3},
        "tokens_per_second": 15,
        "requests_per_minute": 90,
    }
