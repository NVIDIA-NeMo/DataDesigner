# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from httpx_retries import RetryTransport

from data_designer.engine.models.clients.retry import RetryConfig, create_retry_transport


@pytest.mark.parametrize(
    "field,expected",
    [
        ("max_retries", 3),
        ("backoff_factor", 2.0),
        ("backoff_jitter", 0.2),
        ("max_backoff_wait", 60.0),
        ("retryable_status_codes", frozenset({429, 502, 503, 504})),
    ],
    ids=["max_retries", "backoff_factor", "backoff_jitter", "max_backoff_wait", "retryable_status_codes"],
)
def test_retry_config_defaults_match_litellm_router(field: str, expected: object) -> None:
    config = RetryConfig()
    assert getattr(config, field) == expected


def test_retry_config_is_frozen() -> None:
    config = RetryConfig()
    with pytest.raises(AttributeError):
        config.max_retries = 10  # type: ignore[misc]


def test_create_retry_transport_returns_retry_transport() -> None:
    transport = create_retry_transport()
    assert isinstance(transport, RetryTransport)


def test_create_retry_transport_with_none_uses_defaults() -> None:
    transport = create_retry_transport(None)
    assert transport.retry.total == 3


@pytest.mark.parametrize(
    "field,config_value,retry_attr,expected",
    [
        ("max_retries", 5, "total", 5),
        ("backoff_factor", 1.0, "backoff_factor", 1.0),
        ("retryable_status_codes", frozenset({429, 500}), "status_forcelist", frozenset({429, 500})),
    ],
    ids=["max_retries", "backoff_factor", "status_forcelist"],
)
def test_create_retry_transport_propagates_config(
    field: str, config_value: object, retry_attr: str, expected: object
) -> None:
    config = RetryConfig(**{field: config_value})
    transport = create_retry_transport(config)
    assert getattr(transport.retry, retry_attr) == expected
