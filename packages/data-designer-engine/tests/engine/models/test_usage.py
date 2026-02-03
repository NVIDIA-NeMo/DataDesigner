# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.engine.models.usage import ModelUsageStats, RequestUsageStats, TokenUsageStats, ToolUsageStats


def test_token_usage_stats() -> None:
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


def test_request_usage_stats() -> None:
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


def test_tool_usage_stats_empty_state() -> None:
    """Test ToolUsageStats initialization with empty state."""
    tool_usage = ToolUsageStats()
    assert tool_usage.total_tool_calls == 0
    assert tool_usage.total_tool_call_turns == 0
    assert tool_usage.generations_with_tools == 0
    assert tool_usage.has_usage is False
    assert tool_usage.turns_per_generation_mean == 0.0
    assert tool_usage.turns_per_generation_stddev == 0.0
    assert tool_usage.calls_per_generation_mean == 0.0
    assert tool_usage.calls_per_generation_stddev == 0.0


def test_tool_usage_stats_single_generation() -> None:
    """Test ToolUsageStats with a single generation - stddev should be 0."""
    tool_usage = ToolUsageStats()
    tool_usage.extend(tool_calls=5, tool_call_turns=2)

    assert tool_usage.total_tool_calls == 5
    assert tool_usage.total_tool_call_turns == 2
    assert tool_usage.generations_with_tools == 1
    assert tool_usage.has_usage is True
    assert tool_usage.calls_per_generation_mean == 5.0
    assert tool_usage.turns_per_generation_mean == 2.0
    # With single sample, stddev should be 0
    assert tool_usage.calls_per_generation_stddev == 0.0
    assert tool_usage.turns_per_generation_stddev == 0.0


def test_tool_usage_stats_multiple_generations_identical_values() -> None:
    """Test ToolUsageStats with multiple identical generations - stddev should be 0."""
    tool_usage = ToolUsageStats()
    for _ in range(3):
        tool_usage.extend(tool_calls=4, tool_call_turns=3)

    assert tool_usage.total_tool_calls == 12
    assert tool_usage.total_tool_call_turns == 9
    assert tool_usage.generations_with_tools == 3
    assert tool_usage.has_usage is True
    assert tool_usage.calls_per_generation_mean == 4.0
    assert tool_usage.turns_per_generation_mean == 3.0
    # With identical values, stddev should be 0
    assert tool_usage.calls_per_generation_stddev == 0.0
    assert tool_usage.turns_per_generation_stddev == 0.0


def test_tool_usage_stats_multiple_generations_varying_values() -> None:
    """Test ToolUsageStats with varying values - verify mean and stddev calculations."""
    tool_usage = ToolUsageStats()
    tool_usage.extend(tool_calls=2, tool_call_turns=1)
    tool_usage.extend(tool_calls=4, tool_call_turns=3)
    tool_usage.extend(tool_calls=6, tool_call_turns=2)

    assert tool_usage.total_tool_calls == 12
    assert tool_usage.total_tool_call_turns == 6
    assert tool_usage.generations_with_tools == 3
    assert tool_usage.has_usage is True

    # Mean calculations: calls = (2+4+6)/3 = 4.0, turns = (1+3+2)/3 = 2.0
    assert tool_usage.calls_per_generation_mean == 4.0
    assert tool_usage.turns_per_generation_mean == 2.0

    # Stddev calculations (population stddev):
    # calls: variance = (4+16+36)/3 - 16 = 56/3 - 16 = 2.667, stddev = sqrt(2.667) ≈ 1.633
    # turns: variance = (1+9+4)/3 - 4 = 14/3 - 4 = 0.667, stddev = sqrt(0.667) ≈ 0.816
    assert tool_usage.calls_per_generation_stddev == pytest.approx(1.6329931618554521, rel=1e-6)
    assert tool_usage.turns_per_generation_stddev == pytest.approx(0.816496580927726, rel=1e-6)


def test_tool_usage_stats_zero_tool_calls_not_counted() -> None:
    """Test that extend with zero tool_call_turns does not increment generations_with_tools."""
    tool_usage = ToolUsageStats()
    tool_usage.extend(tool_calls=0, tool_call_turns=0)

    assert tool_usage.total_tool_calls == 0
    assert tool_usage.total_tool_call_turns == 0
    assert tool_usage.generations_with_tools == 0
    assert tool_usage.has_usage is False
    assert tool_usage.turns_per_generation_mean == 0.0
    assert tool_usage.turns_per_generation_stddev == 0.0
    assert tool_usage.calls_per_generation_mean == 0.0
    assert tool_usage.calls_per_generation_stddev == 0.0


def test_tool_usage_stats_mixed_zero_and_nonzero_generations() -> None:
    """Test that only generations with tool_call_turns > 0 are counted."""
    tool_usage = ToolUsageStats()
    tool_usage.extend(tool_calls=0, tool_call_turns=0)  # Should not count
    tool_usage.extend(tool_calls=4, tool_call_turns=2)  # Should count
    tool_usage.extend(tool_calls=0, tool_call_turns=0)  # Should not count
    tool_usage.extend(tool_calls=6, tool_call_turns=4)  # Should count

    assert tool_usage.total_tool_calls == 10
    assert tool_usage.total_tool_call_turns == 6
    assert tool_usage.generations_with_tools == 2
    assert tool_usage.has_usage is True

    # Mean: calls = (4+6)/2 = 5.0, turns = (2+4)/2 = 3.0
    assert tool_usage.calls_per_generation_mean == 5.0
    assert tool_usage.turns_per_generation_mean == 3.0

    # Stddev: calls variance = (16+36)/2 - 25 = 26 - 25 = 1, stddev = 1.0
    # Stddev: turns variance = (4+16)/2 - 9 = 10 - 9 = 1, stddev = 1.0
    assert tool_usage.calls_per_generation_stddev == 1.0
    assert tool_usage.turns_per_generation_stddev == 1.0


def test_model_usage_stats() -> None:
    model_usage_stats = ModelUsageStats()
    assert model_usage_stats.token_usage.input_tokens == 0
    assert model_usage_stats.token_usage.output_tokens == 0
    assert model_usage_stats.request_usage.successful_requests == 0
    assert model_usage_stats.request_usage.failed_requests == 0
    assert model_usage_stats.has_usage is False

    # tool_usage is excluded when has_usage is False
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

    # tool_usage is excluded when has_usage is False
    assert model_usage_stats.get_usage_stats(total_time_elapsed=2) == {
        "token_usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        "request_usage": {"successful_requests": 2, "failed_requests": 1, "total_requests": 3},
        "tokens_per_second": 15,
        "requests_per_minute": 90,
    }
