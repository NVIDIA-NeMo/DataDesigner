# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

import litellm
import pytest

from data_designer.engine.models.litellm_overrides import (
    DEFAULT_MAX_CALLBACKS,
    CustomRouter,
    ThreadSafeCache,
    apply_litellm_patches,
)


@pytest.fixture
def stub_thread_safe_cache():
    return ThreadSafeCache()


@pytest.fixture
def stub_custom_router():
    return CustomRouter([], initial_retry_after_s=1.0, jitter_pct=0.0)


@pytest.mark.parametrize(
    "retry_count,jitter,expected_sleep_s",
    [
        (0, 0.0, 1.0),
        (1, 0.0, 2.0),
        (2, 0.0, 4.0),
        (3, 0.0, 8.0),
        (0, 0.2, 1.2),
        (1, 0.2, 2.4),
        (2, 0.2, 4.8),
        (3, 0.2, 9.6),
    ],
)
def test_custom_router_calculate_exponential_backoff(retry_count: int, jitter: float, expected_sleep_s: float):
    with patch("random.uniform", return_value=jitter):
        assert (
            CustomRouter.calculate_exponential_backoff(
                initial_retry_after_s=1, current_retry=retry_count, jitter_pct=jitter
            )
            == expected_sleep_s
        )


def test_apply_litellm_patches_no_exceptions():
    try:
        apply_litellm_patches()
    except Exception as e:
        pytest.fail(f"apply_litellm_patches() raised an unexpected exception: {e}")


@patch("data_designer.engine.models.litellm_overrides.quiet_noisy_logger", autospec=True)
def test_apply_litellm_patches(mock_quiet_noisy_logger):
    apply_litellm_patches()
    assert isinstance(litellm.in_memory_llm_clients_cache, ThreadSafeCache)
    assert (
        litellm.litellm_core_utils.logging_callback_manager.LoggingCallbackManager.MAX_CALLBACKS
        == DEFAULT_MAX_CALLBACKS
    )
    assert mock_quiet_noisy_logger.call_count == 3
    assert mock_quiet_noisy_logger.call_args_list[0][0][0] == "httpx"
    assert mock_quiet_noisy_logger.call_args_list[1][0][0] == "LiteLLM"
    assert mock_quiet_noisy_logger.call_args_list[2][0][0] == "LiteLLM Router"


@pytest.mark.parametrize(
    "test_case,key,value,expected_result",
    [
        ("get_cache", "test_key", "test_value", "test_value"),
        ("set_cache", "test_key", "test_value", "test_value"),
    ],
)
def test_thread_safe_cache_basic_operations(stub_thread_safe_cache, test_case, key, value, expected_result):
    stub_thread_safe_cache.set_cache(key, value)
    result = stub_thread_safe_cache.get_cache(key)
    assert result == expected_result


def test_thread_safe_cache_batch_get_cache(stub_thread_safe_cache):
    stub_thread_safe_cache.set_cache("key1", "value1")
    stub_thread_safe_cache.set_cache("key2", "value2")

    result = stub_thread_safe_cache.batch_get_cache(["key1", "key2"])
    assert result == ["value1", "value2"]


def test_thread_safe_cache_delete_cache(stub_thread_safe_cache):
    stub_thread_safe_cache.set_cache("test_key", "test_value")
    stub_thread_safe_cache.delete_cache("test_key")

    result = stub_thread_safe_cache.get_cache("test_key")
    assert result is None


def test_thread_safe_cache_evict_cache(stub_thread_safe_cache):
    stub_thread_safe_cache.set_cache("test_key", "test_value")
    stub_thread_safe_cache.evict_cache()
    stub_thread_safe_cache.get_cache("test_key")
    assert True


def test_thread_safe_cache_increment_cache(stub_thread_safe_cache):
    stub_thread_safe_cache.set_cache("counter", 5)

    result = stub_thread_safe_cache.increment_cache("counter", 3)
    assert result == 8

    final_value = stub_thread_safe_cache.get_cache("counter")
    assert final_value == 8


def test_thread_safe_cache_flush_cache(stub_thread_safe_cache):
    stub_thread_safe_cache.set_cache("key1", "value1")
    stub_thread_safe_cache.set_cache("key2", "value2")
    stub_thread_safe_cache.flush_cache()

    assert stub_thread_safe_cache.get_cache("key1") is None
    assert stub_thread_safe_cache.get_cache("key2") is None


def test_custom_router_initialization():
    router = CustomRouter([], initial_retry_after_s=2.0, jitter_pct=0.1)

    assert router._initial_retry_after_s == 2.0
    assert router._jitter_pct == 0.1


@patch("random.uniform", return_value=0.1, autospec=True)
def test_custom_router_calculate_exponential_backoff_with_jitter(mock_uniform):
    result = CustomRouter.calculate_exponential_backoff(initial_retry_after_s=1.0, current_retry=2, jitter_pct=0.2)
    assert result >= 4.0
    assert result <= 4.4
    mock_uniform.assert_called_once_with(-0.2, 0.2)
