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

from data_designer.config.run_config import ThrottleConfig

logger = logging.getLogger(__name__)


class ThrottleDomain(str, Enum):
    CHAT = "chat"
    EMBEDDING = "embedding"
    IMAGE = "image"
    HEALTHCHECK = "healthcheck"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MIN_LIMIT: int = 1
DEFAULT_ACQUIRE_TIMEOUT: float = 300.0
CAPACITY_POLL_INTERVAL: float = 0.05


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
    waiters: int = 0
    rate_limit_ceiling: int = 0
    consecutive_429s: int = 0


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
    """Adaptive concurrency manager using AIMD (Additive Increase /
    Multiplicative Decrease).

    Keyed at two levels:

    - **Global cap**: ``(provider_name, model_id)`` — shared hard ceiling.
    - **Domain**: ``(provider_name, model_id, throttle_domain)`` — per-route
      AIMD state that floats between 1 and the global effective max.

    **AIMD behaviour**:

    - *Decrease* — on a 429 / rate-limit signal the domain's concurrency limit
      is multiplied by ``reduce_factor`` (default 0.75, i.e. reduced by 25%)
      and a cooldown block is applied for ``retry_after`` seconds (or
      ``default_cooldown_seconds``).
    - *Increase* — after every ``success_window`` consecutive successful
      releases the limit grows by ``additive_increase`` (default 1), up to
      the *rate-limit ceiling* (or the global effective max if no 429 has
      been observed yet).
    - *Stabilization* — each 429 records the pre-halving limit as
      ``rate_limit_ceiling``.  Subsequent additive increases stop at
      ``ceiling * (1 + ceiling_overshoot)`` (default 10%) instead of
      climbing all the way to ``effective_max``.  The overshoot band lets
      the system probe whether the endpoint can now handle more traffic
      (e.g. after load drops) while dampening the sawtooth.  If the probe
      succeeds, the ceiling ratchets up; if it triggers another 429, the
      ceiling lowers.

    Thread-safe: all state mutations are guarded by a single lock so that
    sync and async callers co-throttle correctly.
    """

    def __init__(
        self,
        config: ThrottleConfig | None = None,
    ) -> None:
        tc = config or ThrottleConfig()
        self._reduce_factor = tc.reduce_factor
        self._additive_increase = tc.additive_increase
        self._success_window = tc.success_window
        self._default_cooldown_seconds = tc.cooldown_seconds
        self._ceiling_overshoot = tc.ceiling_overshoot
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

        **Ordering invariant:** ``register()`` must be called for a
        ``(provider_name, model_id)`` pair *before* any ``try_acquire()`` for
        the same key.  If ``try_acquire()`` runs first it creates a domain at
        ``DEFAULT_MIN_LIMIT`` and ``_clamp_domains`` only *decreases* limits,
        so a later ``register()`` will not raise the domain to the intended
        capacity.
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

    def is_registered(self, provider_name: str, model_id: str) -> bool:
        """Return ``True`` if ``register()`` has been called for this key."""
        with self._lock:
            return (provider_name, model_id) in self._global_caps

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

        Raises ``RuntimeError`` if the ``(provider_name, model_id)`` pair
        has not been registered via ``register()``.
        """
        now = now if now is not None else time.monotonic()
        with self._lock:
            if (provider_name, model_id) not in self._global_caps:
                raise RuntimeError(
                    f"ThrottleManager.try_acquire() called before register() "
                    f"for ({provider_name!r}, {model_id!r}). "
                    f"Call register() first to set the concurrency limit."
                )
            state = self._get_or_create_domain(provider_name, model_id, domain)
            if now < state.blocked_until:
                return state.blocked_until - now
            if state.in_flight >= state.current_limit:
                return CAPACITY_POLL_INTERVAL
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
        with self._lock:
            state = self._get_or_create_domain(provider_name, model_id, domain)
            state.in_flight = max(0, state.in_flight - 1)
            state.consecutive_429s = 0
            state.success_streak += 1
            if state.success_streak >= self._success_window:
                effective_max = self._effective_max_for(provider_name, model_id)
                cap = self._compute_soft_ceiling(state, effective_max)
                if state.current_limit < cap:
                    prev = state.current_limit
                    state.current_limit = min(state.current_limit + self._additive_increase, cap)
                    if state.current_limit >= cap:
                        if cap < effective_max:
                            logger.info(
                                "🔋✅ '%s' [%s] concurrency recovered to %d parallel requests",
                                model_id,
                                domain.value,
                                state.current_limit,
                            )
                        else:
                            logger.info(
                                "🔋✅ '%s' [%s] concurrency fully recovered (%d parallel requests)",
                                model_id,
                                domain.value,
                                state.current_limit,
                            )
                    else:
                        logger.info(
                            "🪫📈🔥 '%s' [%s] concurrency increased from %d → %d",
                            model_id,
                            domain.value,
                            prev,
                            state.current_limit,
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
            is_first_in_cascade = state.consecutive_429s == 0
            state.consecutive_429s += 1
            cooldown_duration = (
                retry_after if retry_after is not None and retry_after > 0 else self._default_cooldown_seconds
            )
            state.blocked_until = now + cooldown_duration
            state.success_streak = 0

            if is_first_in_cascade:
                state.current_limit = max(DEFAULT_MIN_LIMIT, math.floor(state.current_limit * self._reduce_factor))
                if state.current_limit < prev_limit:
                    if state.rate_limit_ceiling == 0:
                        state.rate_limit_ceiling = prev_limit
                    else:
                        state.rate_limit_ceiling = min(state.rate_limit_ceiling, prev_limit)
                    if state.rate_limit_ceiling < prev_limit:
                        logger.info(
                            "🪫📉 '%s' [%s] server rate-limited at %d (server limit ~%d) — concurrency reduced to %d (retrying in %.0fs)",
                            model_id,
                            domain.value,
                            prev_limit,
                            state.rate_limit_ceiling,
                            state.current_limit,
                            cooldown_duration,
                        )
                    else:
                        logger.info(
                            "🪫📉 '%s' [%s] server rate-limited — concurrency reduced from %d → %d (retrying in %.0fs)",
                            model_id,
                            domain.value,
                            prev_limit,
                            state.current_limit,
                            cooldown_duration,
                        )
            else:
                logger.debug(
                    "Throttle %s [%s] cascade 429 #%d (limit held at %d)",
                    model_id,
                    domain.value,
                    state.consecutive_429s,
                    state.current_limit,
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
        timeout: float = DEFAULT_ACQUIRE_TIMEOUT,
    ) -> None:
        deadline = time.monotonic() + timeout
        wait = self.try_acquire(provider_name=provider_name, model_id=model_id, domain=domain)
        if wait == 0.0:
            return
        with self._lock:
            state = self._get_or_create_domain(provider_name, model_id, domain)
            state.waiters += 1
            if state.waiters == 1:
                logger.debug(
                    "Throttle %s/%s [%s] queue forming (in_flight=%d/%d)",
                    provider_name,
                    model_id,
                    domain.value,
                    state.in_flight,
                    state.current_limit,
                )
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or wait > remaining:
                    raise TimeoutError(
                        f"Throttle acquire timed out after {timeout:.0f}s "
                        f"for {provider_name}/{model_id} [{domain.value}]"
                    )
                time.sleep(min(wait, remaining))
                wait = self.try_acquire(provider_name=provider_name, model_id=model_id, domain=domain)
                if wait == 0.0:
                    return
        finally:
            with self._lock:
                state.waiters -= 1
                if state.waiters == 0:
                    logger.debug(
                        "Throttle %s/%s [%s] queue drained",
                        provider_name,
                        model_id,
                        domain.value,
                    )

    async def acquire_async(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
        timeout: float = DEFAULT_ACQUIRE_TIMEOUT,
    ) -> None:
        deadline = time.monotonic() + timeout
        wait = self.try_acquire(provider_name=provider_name, model_id=model_id, domain=domain)
        if wait == 0.0:
            return
        with self._lock:
            state = self._get_or_create_domain(provider_name, model_id, domain)
            state.waiters += 1
            if state.waiters == 1:
                logger.debug(
                    "Throttle %s/%s [%s] queue forming (in_flight=%d/%d)",
                    provider_name,
                    model_id,
                    domain.value,
                    state.in_flight,
                    state.current_limit,
                )
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or wait > remaining:
                    raise TimeoutError(
                        f"Throttle acquire timed out after {timeout:.0f}s "
                        f"for {provider_name}/{model_id} [{domain.value}]"
                    )
                await asyncio.sleep(min(wait, remaining))
                wait = self.try_acquire(provider_name=provider_name, model_id=model_id, domain=domain)
                if wait == 0.0:
                    return
        finally:
            with self._lock:
                state.waiters -= 1
                if state.waiters == 0:
                    logger.debug(
                        "Throttle %s/%s [%s] queue drained",
                        provider_name,
                        model_id,
                        domain.value,
                    )

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

    def _compute_soft_ceiling(self, state: DomainThrottleState, effective_max: int) -> int:
        """Return the upper bound for additive increase.

        If a rate-limit ceiling has been recorded, allow probing up to
        ``ceiling * (1 + overshoot)`` (clamped to ``effective_max``).
        Otherwise fall back to ``effective_max``.
        """
        if state.rate_limit_ceiling <= 0:
            return effective_max
        soft = state.rate_limit_ceiling + max(1, math.floor(state.rate_limit_ceiling * self._ceiling_overshoot))
        return min(soft, effective_max)

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
