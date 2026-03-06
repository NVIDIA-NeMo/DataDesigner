# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ThrottleDomain(str, Enum):
    CHAT = "chat"
    EMBEDDING = "embedding"
    IMAGE = "image"
    HEALTHCHECK = "healthcheck"


# ---------------------------------------------------------------------------
# AIMD tuning defaults
# ---------------------------------------------------------------------------

DEFAULT_REDUCE_FACTOR: float = 0.5
DEFAULT_SUCCESS_WINDOW: int = 50
DEFAULT_BLOCK_SECONDS: float = 2.0
DEFAULT_MIN_LIMIT: int = 1


# ---------------------------------------------------------------------------
# Internal state containers
# ---------------------------------------------------------------------------


@dataclass
class DomainThrottleState:
    """Per-domain AIMD concurrency state.

    All mutations must be performed while holding the owning
    ``ThrottleManager._lock``.
    """

    current_limit: int
    in_flight: int = 0
    blocked_until: float = 0.0
    success_streak: int = 0


@dataclass
class GlobalCapState:
    """Tracks the effective hard cap across aliases sharing a provider+model."""

    limits_by_alias: dict[str, int] = field(default_factory=dict)
    effective_max: int = 0

    def register_alias(self, alias: str, max_parallel: int) -> None:
        self.limits_by_alias[alias] = max_parallel
        self.effective_max = min(self.limits_by_alias.values())


# ---------------------------------------------------------------------------
# ThrottleManager
# ---------------------------------------------------------------------------


class ThrottleManager:
    """Adaptive concurrency manager using AIMD.

    Keyed at two levels:
    - **Global cap**: ``(provider_name, model_id)`` — shared hard ceiling.
    - **Domain**: ``(provider_name, model_id, throttle_domain)`` — per-route
      AIMD state that floats between 1 and the global effective max.

    Thread-safe: all state mutations are guarded by a single lock so that
    sync and async callers co-throttle correctly.
    """

    def __init__(
        self,
        *,
        reduce_factor: float = DEFAULT_REDUCE_FACTOR,
        success_window: int = DEFAULT_SUCCESS_WINDOW,
        default_block_seconds: float = DEFAULT_BLOCK_SECONDS,
    ) -> None:
        self._reduce_factor = reduce_factor
        self._success_window = success_window
        self._default_block_seconds = default_block_seconds
        self._lock = threading.Lock()
        self._global_caps: dict[tuple[str, str], GlobalCapState] = {}
        self._domains: dict[tuple[str, str, str], DomainThrottleState] = {}

    # -------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------

    def register(
        self,
        *,
        provider_name: str,
        model_id: str,
        alias: str,
        max_parallel_requests: int,
    ) -> None:
        """Register a model alias and its concurrency limit.

        If multiple aliases share the same ``(provider_name, model_id)`` the
        effective max is ``min()`` of all registered limits.  Existing domain
        states are clamped to the new effective max.
        """
        with self._lock:
            global_key = (provider_name, model_id)
            cap = self._global_caps.setdefault(global_key, GlobalCapState())
            cap.register_alias(alias, max_parallel_requests)
            self._clamp_domains(global_key, cap.effective_max)
            logger.debug(
                "Throttle registered alias=%r for %s/%s (max_parallel=%d, effective_max=%d)",
                alias,
                provider_name,
                model_id,
                max_parallel_requests,
                cap.effective_max,
            )

    # -------------------------------------------------------------------
    # Core non-blocking primitives
    # -------------------------------------------------------------------

    def try_acquire(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
        now: float | None = None,
    ) -> float:
        """Attempt to acquire a concurrency slot.

        Returns ``0.0`` if the slot was acquired, otherwise the number of
        seconds the caller should wait before retrying.
        """
        now = now if now is not None else time.monotonic()
        with self._lock:
            state = self._get_or_create_domain(provider_name, model_id, domain)
            if now < state.blocked_until:
                wait = state.blocked_until - now
                logger.debug(
                    "Throttle %s/%s [%s] blocked for %.1fs (cooldown)",
                    provider_name,
                    model_id,
                    domain.value,
                    wait,
                )
                return wait
            if state.in_flight >= state.current_limit:
                logger.debug(
                    "Throttle %s/%s [%s] at capacity (%d/%d), backing off %.1fs",
                    provider_name,
                    model_id,
                    domain.value,
                    state.in_flight,
                    state.current_limit,
                    self._default_block_seconds,
                )
                return self._default_block_seconds
            state.in_flight += 1
            return 0.0

    def release_success(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
        now: float | None = None,
    ) -> None:
        now = now if now is not None else time.monotonic()
        with self._lock:
            state = self._get_or_create_domain(provider_name, model_id, domain)
            state.in_flight = max(0, state.in_flight - 1)
            state.success_streak += 1
            if state.success_streak >= self._success_window:
                effective_max = self._effective_max_for(provider_name, model_id)
                if state.current_limit < effective_max:
                    state.current_limit += 1
                    if state.current_limit >= effective_max:
                        logger.info(
                            "🟢 Throttle %s/%s [%s] recovered to full capacity (%d)",
                            provider_name,
                            model_id,
                            domain.value,
                            state.current_limit,
                        )
                    else:
                        logger.debug(
                            "Throttle %s/%s [%s] AIMD increase: limit %d -> %d (max %d)",
                            provider_name,
                            model_id,
                            domain.value,
                            state.current_limit - 1,
                            state.current_limit,
                            effective_max,
                        )
                state.success_streak = 0

    def release_rate_limited(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
        retry_after: float | None = None,
        now: float | None = None,
    ) -> None:
        now = now if now is not None else time.monotonic()
        with self._lock:
            state = self._get_or_create_domain(provider_name, model_id, domain)
            state.in_flight = max(0, state.in_flight - 1)
            prev_limit = state.current_limit
            state.current_limit = max(DEFAULT_MIN_LIMIT, math.floor(state.current_limit * self._reduce_factor))
            block_duration = retry_after if retry_after and retry_after > 0 else self._default_block_seconds
            state.blocked_until = now + block_duration
            state.success_streak = 0
            logger.warning(
                "🚦 Throttle %s/%s [%s] rate-limited: limit %d -> %d, blocked for %.1fs%s",
                provider_name,
                model_id,
                domain.value,
                prev_limit,
                state.current_limit,
                block_duration,
                f" (retry-after={retry_after:.1f}s)" if retry_after else "",
            )

    def release_failure(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
        now: float | None = None,
    ) -> None:
        with self._lock:
            state = self._get_or_create_domain(provider_name, model_id, domain)
            state.in_flight = max(0, state.in_flight - 1)

    # -------------------------------------------------------------------
    # Sync / async wrappers
    # -------------------------------------------------------------------

    def acquire_sync(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
    ) -> None:
        while True:
            wait = self.try_acquire(provider_name=provider_name, model_id=model_id, domain=domain)
            if wait == 0.0:
                return
            time.sleep(wait)

    async def acquire_async(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
    ) -> None:
        while True:
            wait = self.try_acquire(provider_name=provider_name, model_id=model_id, domain=domain)
            if wait == 0.0:
                return
            await asyncio.sleep(wait)

    # -------------------------------------------------------------------
    # Introspection (useful for tests and telemetry)
    # -------------------------------------------------------------------

    def get_domain_state(
        self,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
    ) -> DomainThrottleState | None:
        with self._lock:
            return self._domains.get((provider_name, model_id, domain.value))

    def get_effective_max(self, provider_name: str, model_id: str) -> int:
        with self._lock:
            return self._effective_max_for(provider_name, model_id)

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _get_or_create_domain(
        self,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
    ) -> DomainThrottleState:
        key = (provider_name, model_id, domain.value)
        state = self._domains.get(key)
        if state is None:
            effective_max = self._effective_max_for(provider_name, model_id)
            state = DomainThrottleState(current_limit=effective_max)
            self._domains[key] = state
        return state

    def _effective_max_for(self, provider_name: str, model_id: str) -> int:
        cap = self._global_caps.get((provider_name, model_id))
        if cap is None or cap.effective_max <= 0:
            return DEFAULT_MIN_LIMIT
        return cap.effective_max

    def _clamp_domains(self, global_key: tuple[str, str], effective_max: int) -> None:
        provider_name, model_id = global_key
        for (pn, mid, _dom), state in self._domains.items():
            if pn == provider_name and mid == model_id and state.current_limit > effective_max:
                state.current_limit = effective_max
