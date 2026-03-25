# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from httpx_retries import Retry, RetryTransport

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)

# 429 must not be retried at the transport layer so that rate-limit signals
# propagate to ThrottledModelClient for AIMD backoff.
_RESERVED_STATUS_CODES: frozenset[int] = frozenset({429})


@dataclass(frozen=True)
class RetryConfig:
    """Retry policy for native HTTP adapters.

    Retries non-rate-limit transient failures (``502``, ``503``, ``504``) and
    connection/transport errors.  ``429`` is intentionally excluded so that
    rate-limit signals reach the ``ThrottledModelClient`` wrapper for AIMD
    backoff.  If a caller includes ``429`` in ``retryable_status_codes``,
    ``create_retry_transport`` will strip it and log a warning.
    """

    max_retries: int = 3
    backoff_factor: float = 2.0
    backoff_jitter: float = 0.2
    max_backoff_wait: float = 60.0
    retryable_status_codes: frozenset[int] = field(default_factory=lambda: frozenset({502, 503, 504}))


def create_retry_transport(
    config: RetryConfig | None = None,
    *,
    strip_rate_limit_codes: bool = True,
    transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
) -> RetryTransport:
    """Build an httpx ``RetryTransport`` from a :class:`RetryConfig`.

    The returned transport handles both sync and async requests (``RetryTransport``
    inherits from ``httpx.BaseTransport`` and ``httpx.AsyncBaseTransport``).

    Args:
        config: Retry policy.  Uses ``RetryConfig()`` defaults when ``None``.
        strip_rate_limit_codes: When ``True`` (default, used by the async engine),
            status codes in ``_RESERVED_STATUS_CODES`` (currently ``{429}``) are
            stripped so that rate-limit responses reach the ``ThrottledModelClient``
            AIMD feedback loop.  When ``False`` (used by the sync engine, which has
            no salvage queue), 429 is kept in the retry list so the transport layer
            retries it transparently.
        transport: Optional pre-configured inner transport to pass directly to
            ``RetryTransport``.  Pass ``httpx.HTTPTransport`` for sync clients or
            ``httpx.AsyncHTTPTransport`` for async clients — typically with a custom
            ``limits=`` — so that the connection pool is sized correctly.  When
            ``None`` (default), ``RetryTransport`` creates its own default pools for
            both sync and async requests.
    """
    cfg = config or RetryConfig()
    status_codes = cfg.retryable_status_codes
    if strip_rate_limit_codes:
        reserved_overlap = status_codes & _RESERVED_STATUS_CODES
        if reserved_overlap:
            logger.warning(
                "Stripping reserved status codes %s from retryable_status_codes; "
                "these must reach ThrottledModelClient for AIMD backoff.",
                sorted(reserved_overlap),
            )
            status_codes = status_codes - _RESERVED_STATUS_CODES
    retry = Retry(
        total=cfg.max_retries,
        backoff_factor=cfg.backoff_factor,
        backoff_jitter=cfg.backoff_jitter,
        max_backoff_wait=cfg.max_backoff_wait,
        status_forcelist=status_codes,
        respect_retry_after_header=True,
        allowed_methods=Retry.RETRYABLE_METHODS | frozenset(["POST"]),
    )
    return RetryTransport(transport=transport, retry=retry)
