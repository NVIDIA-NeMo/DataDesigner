# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

from httpx_retries import Retry, RetryTransport


@dataclass(frozen=True)
class RetryConfig:
    """Retry policy for native HTTP adapters.

    Defaults mirror the current LiteLLM router settings in
    ``LiteLLMRouterDefaultKwargs`` so behavior is preserved during migration.
    """

    max_retries: int = 3
    backoff_factor: float = 2.0
    backoff_jitter: float = 0.2
    max_backoff_wait: float = 60.0
    # TODO: Remove 429 from retryable_status_codes once ThrottleManager is
    # wired via AsyncTaskScheduler (plan 346), so every rate-limit signal
    # reaches AIMD backoff instead of being silently retried at the transport layer.
    retryable_status_codes: frozenset[int] = field(default_factory=lambda: frozenset({429, 502, 503, 504}))


def create_retry_transport(config: RetryConfig | None = None) -> RetryTransport:
    """Build an httpx ``RetryTransport`` from a :class:`RetryConfig`.

    The returned transport handles both sync and async requests (``RetryTransport``
    inherits from ``httpx.BaseTransport`` and ``httpx.AsyncBaseTransport``).
    """
    cfg = config or RetryConfig()
    retry = Retry(
        total=cfg.max_retries,
        backoff_factor=cfg.backoff_factor,
        backoff_jitter=cfg.backoff_jitter,
        max_backoff_wait=cfg.max_backoff_wait,
        status_forcelist=cfg.retryable_status_codes,
        respect_retry_after_header=True,
        allowed_methods=Retry.RETRYABLE_METHODS | frozenset(["POST"]),
    )
    return RetryTransport(retry=retry)
