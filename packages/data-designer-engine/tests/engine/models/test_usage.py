# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.engine.models.usage import (
    ImageUsageStats,
    ModelUsageStats,
    RequestUsageStats,
    TokenUsageStats,
    ToolUsageStats,
)


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


def test_image_usage_stats() -> None:
    image_usage_stats = ImageUsageStats()
    assert image_usage_stats.total_images == 0
    assert image_usage_stats.has_usage is False

    image_usage_stats.extend(images=5)
    assert image_usage_stats.total_images == 5
    assert image_usage_stats.has_usage is True

    image_usage_stats.extend(images=3)
    assert image_usage_stats.total_images == 8
    assert image_usage_stats.has_usage is True


def test_tool_usage_stats_empty_state() -> None:
    """Test ToolUsageStats initialization with empty state."""
    tool_usage = ToolUsageStats()
    assert tool_usage.total_tool_calls == 0
    assert tool_usage.total_tool_call_turns == 0
    assert tool_usage.total_generations == 0
    assert tool_usage.generations_with_tools == 0
    assert tool_usage.has_usage is False


def test_tool_usage_stats_single_generation_with_tools() -> None:
    """Test ToolUsageStats with a single generation that uses tools."""
    tool_usage = ToolUsageStats()
    tool_usage.extend(tool_calls=5, tool_call_turns=2)

    assert tool_usage.total_tool_calls == 5
    assert tool_usage.total_tool_call_turns == 2
    assert tool_usage.total_generations == 1
    assert tool_usage.generations_with_tools == 1
    assert tool_usage.has_usage is True


def test_tool_usage_stats_multiple_generations() -> None:
    """Test ToolUsageStats with multiple generations."""
    tool_usage = ToolUsageStats()
    for _ in range(3):
        tool_usage.extend(tool_calls=4, tool_call_turns=3)

    assert tool_usage.total_tool_calls == 12
    assert tool_usage.total_tool_call_turns == 9
    assert tool_usage.total_generations == 3
    assert tool_usage.generations_with_tools == 3
    assert tool_usage.has_usage is True


def test_tool_usage_stats_generation_without_tool_calls() -> None:
    """Test that extend with zero tool_calls still increments total_generations but not generations_with_tools."""
    tool_usage = ToolUsageStats()
    tool_usage.extend(tool_calls=0, tool_call_turns=0)

    assert tool_usage.total_tool_calls == 0
    assert tool_usage.total_tool_call_turns == 0
    assert tool_usage.total_generations == 1
    assert tool_usage.generations_with_tools == 0
    assert tool_usage.has_usage is True


def test_tool_usage_stats_mixed_generations() -> None:
    """Test ratio tracking with mix of generations with and without tools."""
    tool_usage = ToolUsageStats()
    tool_usage.extend(tool_calls=0, tool_call_turns=0)  # No tools used
    tool_usage.extend(tool_calls=4, tool_call_turns=2)  # Tools used
    tool_usage.extend(tool_calls=0, tool_call_turns=0)  # No tools used
    tool_usage.extend(tool_calls=6, tool_call_turns=4)  # Tools used

    assert tool_usage.total_tool_calls == 10
    assert tool_usage.total_tool_call_turns == 6
    assert tool_usage.total_generations == 4
    assert tool_usage.generations_with_tools == 2
    assert tool_usage.has_usage is True


def test_tool_usage_stats_merge() -> None:
    """Test that merging two ToolUsageStats objects works correctly."""
    stats1 = ToolUsageStats()
    stats1.extend(tool_calls=2, tool_call_turns=1)
    stats1.extend(tool_calls=4, tool_call_turns=3)

    stats2 = ToolUsageStats()
    stats2.extend(tool_calls=6, tool_call_turns=2)
    stats2.extend(tool_calls=0, tool_call_turns=0)  # No tools

    stats1.merge(stats2)

    assert stats1.total_tool_calls == 12
    assert stats1.total_tool_call_turns == 6
    assert stats1.total_generations == 4
    assert stats1.generations_with_tools == 3


def test_tool_usage_stats_merge_empty() -> None:
    """Test merging an empty ToolUsageStats doesn't change values."""
    stats1 = ToolUsageStats()
    stats1.extend(tool_calls=4, tool_call_turns=2)

    stats2 = ToolUsageStats()
    stats1.merge(stats2)

    assert stats1.total_tool_calls == 4
    assert stats1.total_tool_call_turns == 2
    assert stats1.total_generations == 1
    assert stats1.generations_with_tools == 1


def test_model_usage_stats() -> None:
    model_usage_stats = ModelUsageStats()
    assert model_usage_stats.token_usage.input_tokens == 0
    assert model_usage_stats.token_usage.output_tokens == 0
    assert model_usage_stats.request_usage.successful_requests == 0
    assert model_usage_stats.request_usage.failed_requests == 0
    assert model_usage_stats.image_usage.total_images == 0
    assert model_usage_stats.has_usage is False

    # tool_usage and image_usage are excluded when has_usage is False
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

    # tool_usage and image_usage are excluded when has_usage is False
    assert model_usage_stats.get_usage_stats(total_time_elapsed=2) == {
        "token_usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        "request_usage": {"successful_requests": 2, "failed_requests": 1, "total_requests": 3},
        "tokens_per_second": 15,
        "requests_per_minute": 90,
    }


def test_model_usage_stats_extend_with_tool_usage() -> None:
    """Test that ModelUsageStats.extend properly merges tool usage."""
    stats1 = ModelUsageStats()
    stats1.tool_usage.extend(tool_calls=2, tool_call_turns=1)
    stats1.tool_usage.extend(tool_calls=4, tool_call_turns=3)

    stats2 = ModelUsageStats()
    stats2.tool_usage.extend(tool_calls=6, tool_call_turns=2)
    stats2.tool_usage.extend(tool_calls=0, tool_call_turns=0)  # No tools used

    stats1.extend(tool_usage=stats2.tool_usage)

    assert stats1.tool_usage.total_tool_calls == 12
    assert stats1.tool_usage.total_tool_call_turns == 6
    assert stats1.tool_usage.total_generations == 4
    assert stats1.tool_usage.generations_with_tools == 3


def test_model_usage_stats_with_image_usage() -> None:
    """Test that ModelUsageStats includes image_usage when it has usage."""
    model_usage_stats = ModelUsageStats()
    model_usage_stats.extend(
        token_usage=TokenUsageStats(input_tokens=10, output_tokens=20),
        request_usage=RequestUsageStats(successful_requests=1, failed_requests=0),
        image_usage=ImageUsageStats(total_images=5),
    )

    assert model_usage_stats.image_usage.total_images == 5
    assert model_usage_stats.image_usage.has_usage is True

    # image_usage should be included in output
    usage_stats = model_usage_stats.get_usage_stats(total_time_elapsed=2)
    assert "image_usage" in usage_stats
    assert usage_stats["image_usage"] == {"total_images": 5}


def test_model_usage_stats_exclude_unused_stats() -> None:
    """Test that ModelUsageStats excludes tool_usage and image_usage when they have no usage."""
    model_usage_stats = ModelUsageStats()
    model_usage_stats.extend(
        token_usage=TokenUsageStats(input_tokens=10, output_tokens=20),
        request_usage=RequestUsageStats(successful_requests=1, failed_requests=0),
    )

    usage_stats = model_usage_stats.get_usage_stats(total_time_elapsed=2)
    assert "tool_usage" not in usage_stats
    assert "image_usage" not in usage_stats
    assert "token_usage" in usage_stats
    assert "request_usage" in usage_stats
