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


def test_tool_usage_stats_merge() -> None:
    """Test that merging two ToolUsageStats objects preserves stddev accuracy."""
    # Create first stats with varying values
    stats1 = ToolUsageStats()
    stats1.extend(tool_calls=2, tool_call_turns=1)
    stats1.extend(tool_calls=4, tool_call_turns=3)

    # Create second stats with varying values
    stats2 = ToolUsageStats()
    stats2.extend(tool_calls=6, tool_call_turns=2)

    # Merge stats2 into stats1
    stats1.merge(stats2)

    # Should have same values as if all three extends were on one object
    assert stats1.total_tool_calls == 12
    assert stats1.total_tool_call_turns == 6
    assert stats1.generations_with_tools == 3

    # Mean calculations should be same as test_tool_usage_stats_multiple_generations_varying_values
    assert stats1.calls_per_generation_mean == 4.0
    assert stats1.turns_per_generation_mean == 2.0

    # Stddev should be same as test_tool_usage_stats_multiple_generations_varying_values
    assert stats1.calls_per_generation_stddev == pytest.approx(1.6329931618554521, rel=1e-6)
    assert stats1.turns_per_generation_stddev == pytest.approx(0.816496580927726, rel=1e-6)


def test_tool_usage_stats_merge_empty() -> None:
    """Test merging an empty ToolUsageStats doesn't change values."""
    stats1 = ToolUsageStats()
    stats1.extend(tool_calls=4, tool_call_turns=2)

    stats2 = ToolUsageStats()
    stats1.merge(stats2)

    assert stats1.total_tool_calls == 4
    assert stats1.total_tool_call_turns == 2
    assert stats1.generations_with_tools == 1
    assert stats1.calls_per_generation_mean == 4.0
    assert stats1.turns_per_generation_mean == 2.0


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


def test_model_usage_stats_extend_preserves_tool_usage_stddev() -> None:
    """Test that ModelUsageStats.extend properly preserves tool usage stddev accuracy."""
    # Create first model stats with tool usage
    stats1 = ModelUsageStats()
    stats1.tool_usage.extend(tool_calls=2, tool_call_turns=1)
    stats1.tool_usage.extend(tool_calls=4, tool_call_turns=3)

    # Create second model stats with tool usage
    stats2 = ModelUsageStats()
    stats2.tool_usage.extend(tool_calls=6, tool_call_turns=2)

    # Extend stats1 with stats2
    stats1.extend(tool_usage=stats2.tool_usage)

    # Should have same results as merging directly
    assert stats1.tool_usage.total_tool_calls == 12
    assert stats1.tool_usage.total_tool_call_turns == 6
    assert stats1.tool_usage.generations_with_tools == 3
    assert stats1.tool_usage.calls_per_generation_mean == 4.0
    assert stats1.tool_usage.turns_per_generation_mean == 2.0
    assert stats1.tool_usage.calls_per_generation_stddev == pytest.approx(1.6329931618554521, rel=1e-6)
    assert stats1.tool_usage.turns_per_generation_stddev == pytest.approx(0.816496580927726, rel=1e-6)


def test_tool_usage_stats_delta_returns_nan_for_stddev() -> None:
    """Test that delta objects (created without sum of squares) return NaN for stddev."""
    import math

    # Create a delta object directly from counts (simulating get_tool_usage_delta)
    delta = ToolUsageStats(
        total_tool_calls=10,
        total_tool_call_turns=5,
        generations_with_tools=3,
    )

    # Mean should still be computable
    assert delta.calls_per_generation_mean == pytest.approx(10 / 3)
    assert delta.turns_per_generation_mean == pytest.approx(5 / 3)

    # Stddev should be NaN since sum of squares wasn't tracked
    assert math.isnan(delta.calls_per_generation_stddev)
    assert math.isnan(delta.turns_per_generation_stddev)


def test_tool_usage_stats_empty_delta_returns_zero_for_stddev() -> None:
    """Test that empty delta objects return 0 for stddev (not NaN)."""
    # Empty delta has no generations, so 0.0 is appropriate
    delta = ToolUsageStats(
        total_tool_calls=0,
        total_tool_call_turns=0,
        generations_with_tools=0,
    )

    assert delta.calls_per_generation_stddev == 0.0
    assert delta.turns_per_generation_stddev == 0.0
