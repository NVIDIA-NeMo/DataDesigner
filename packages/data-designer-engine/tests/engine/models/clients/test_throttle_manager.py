# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from data_designer.config.run_config import ThrottleConfig
from data_designer.engine.models.clients.throttle_manager import (
    CAPACITY_POLL_INTERVAL,
    ThrottleDomain,
    ThrottleManager,
)

PROVIDER = "test-provider"
MODEL = "gpt-test"
DOMAIN = ThrottleDomain.CHAT


@pytest.fixture
def manager() -> ThrottleManager:
    tm = ThrottleManager()
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=4)
    return tm


# --- try_acquire ---


def test_acquire_under_limit_returns_zero(manager: ThrottleManager) -> None:
    wait = manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    assert wait == 0.0


def test_acquire_at_capacity_returns_short_poll_interval(manager: ThrottleManager) -> None:
    for _ in range(4):
        manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    wait = manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    assert wait == pytest.approx(CAPACITY_POLL_INTERVAL)


def test_acquire_respects_blocked_until(manager: ThrottleManager) -> None:
    """Rate-limit cooldown returns remaining block duration (not the short capacity poll)."""
    manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    manager.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, retry_after=5.0, now=1.0)
    wait = manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=2.0)
    assert wait == pytest.approx(4.0, abs=0.01)


def test_acquire_without_registration_raises() -> None:
    tm = ThrottleManager()
    with pytest.raises(RuntimeError, match="register"):
        tm.try_acquire(provider_name="unknown", model_id="m", domain=DOMAIN, now=0.0)


# --- startup ramp ---


def test_startup_ramp_starts_at_one_and_reaches_effective_max_linearly() -> None:
    tm = ThrottleManager(ThrottleConfig(rampup_seconds=10.0))
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=5)

    assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0) == 0.0
    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.current_limit == 1
    assert state.rampup_active is True

    assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0) == pytest.approx(
        CAPACITY_POLL_INTERVAL
    )
    tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    assert state.success_streak == 0

    assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=5.0) == 0.0
    assert state.current_limit == 3
    assert state.rampup_active is True
    tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=5.0)

    assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=10.0) == 0.0
    assert state.current_limit == 5
    assert state.rampup_active is False


def test_rate_limit_aborts_startup_ramp_and_continues_with_aimd() -> None:
    tm = ThrottleManager(ThrottleConfig(reduce_factor=0.5, success_window=1, rampup_seconds=100.0))
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=9)

    assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0) == 0.0
    tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=50.0) == 0.0
    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.current_limit == 5
    assert state.rampup_active is True

    tm.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=50.0)
    assert state.rampup_active is False
    assert state.current_limit == 2
    assert state.rate_limit_ceiling == 5

    assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=60.0) == 0.0
    tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=60.0)
    assert state.current_limit == 3


def test_rate_limit_at_start_of_ramp_does_not_pin_recovery_to_minimum_ceiling() -> None:
    tm = ThrottleManager(
        ThrottleConfig(reduce_factor=0.5, success_window=1, ceiling_overshoot=0.0, rampup_seconds=100.0)
    )
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=10)

    assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0) == 0.0
    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.current_limit == 1

    tm.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    assert state.rampup_active is False
    assert state.current_limit == 1
    assert state.rate_limit_ceiling == 0

    assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=10.0) == 0.0
    tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=10.0)
    assert state.current_limit == 2

    assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=11.0) == 0.0
    tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=11.0)
    assert state.current_limit == 3


def test_startup_ramp_skipped_when_effective_max_is_one() -> None:
    tm = ThrottleManager(ThrottleConfig(rampup_seconds=10.0))
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=1)

    assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0) == 0.0
    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.current_limit == 1
    assert state.rampup_active is False


def test_startup_ramp_completes_on_first_call_after_elapsed_time() -> None:
    tm = ThrottleManager(ThrottleConfig(rampup_seconds=10.0))
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=5)

    assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0) == 0.0
    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.current_limit == 1
    assert state.rampup_active is True
    tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)

    assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=11.0) == 0.0
    assert state.current_limit == 5
    assert state.rampup_active is False


def test_release_failure_preserves_startup_ramp_and_progress() -> None:
    tm = ThrottleManager(ThrottleConfig(rampup_seconds=10.0))
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=5)

    assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0) == 0.0
    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.rampup_active is True
    tm.release_failure(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    assert state.rampup_active is True

    assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=5.0) == 0.0
    assert state.current_limit == 3
    assert state.rampup_active is True


def test_non_ramp_rate_limit_at_minimum_does_not_pin_recovery_to_soft_ceiling() -> None:
    tm = ThrottleManager(ThrottleConfig(reduce_factor=0.5, success_window=1, ceiling_overshoot=0.0))
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=10)

    assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0) == 0.0
    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    state.current_limit = 1

    tm.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    assert state.current_limit == 1
    assert state.rate_limit_ceiling == 0

    assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=10.0) == 0.0
    tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=10.0)
    assert state.current_limit == 2

    assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=11.0) == 0.0
    tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=11.0)
    assert state.current_limit == 3


# --- release_success ---


def test_release_success_frees_slot(manager: ThrottleManager) -> None:
    for _ in range(4):
        manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    manager.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    wait = manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    assert wait == 0.0


def test_additive_increase_after_success_window() -> None:
    tm = ThrottleManager(ThrottleConfig(success_window=5))
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=10)
    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    tm.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    limit_after_drop = state.current_limit

    for i in range(5):
        tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=float(i))
        tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=float(i))

    assert state.current_limit == limit_after_drop + 1


def test_additive_increase_uses_configured_step() -> None:
    tm = ThrottleManager(ThrottleConfig(success_window=1, additive_increase=3))
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=20)
    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    tm.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    limit_after_drop = state.current_limit

    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=1.0)
    tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=1.0)

    assert state.current_limit == limit_after_drop + 3


def test_current_limit_never_exceeds_effective_max() -> None:
    tm = ThrottleManager(ThrottleConfig(success_window=1))
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=2)
    for i in range(20):
        tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=float(i))
        tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=float(i))
    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.current_limit <= 2


def test_additive_increase_clamped_to_effective_max() -> None:
    tm = ThrottleManager(ThrottleConfig(success_window=1, additive_increase=100))
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=5)
    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    tm.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)

    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=1.0)
    tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=1.0)

    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.current_limit == 5


# --- release_rate_limited ---


def test_rate_limited_reduces_current_limit(manager: ThrottleManager) -> None:
    manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    manager.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    state = manager.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.current_limit == 3  # floor(4 * 0.75)


def test_rate_limited_never_drops_below_one() -> None:
    tm = ThrottleManager()
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=1)
    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    tm.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.current_limit >= 1


def test_rate_limited_resets_success_streak(manager: ThrottleManager) -> None:
    manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    manager.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    manager.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    state = manager.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.success_streak == 0


def test_rate_limited_uses_retry_after_for_blocked_until(manager: ThrottleManager) -> None:
    manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=10.0)
    manager.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, retry_after=7.0, now=10.0)
    state = manager.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.blocked_until == pytest.approx(17.0, abs=0.01)


def test_rate_limited_uses_default_block_when_no_retry_after(manager: ThrottleManager) -> None:
    manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=10.0)
    manager.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=10.0)
    state = manager.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.blocked_until == pytest.approx(10.0 + ThrottleConfig.DEFAULT_COOLDOWN_SECONDS, abs=0.01)


# --- release_failure ---


def test_failure_releases_slot_without_limit_change(manager: ThrottleManager) -> None:
    manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    state = manager.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    limit_before = state.current_limit
    manager.release_failure(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    assert state.current_limit == limit_before
    assert state.in_flight == 0


def test_failure_does_not_reset_cascade_while_burst_in_flight(manager: ThrottleManager) -> None:
    """Mixed-response burst (429 → 500 → 429 with multiple slots in-flight) must reduce only once.

    With a real burst of in-flight requests, an interleaved non-rate-limit
    failure should NOT break the cascade - otherwise the next 429 from the
    same wave would be treated as a new cascade and double-reduce the limit
    even though the provider hasn't recovered between the two 429s.
    """
    # Saturate to limit (4 concurrent slots).
    for _ in range(4):
        manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    state = manager.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.in_flight == 4
    limit_before = state.current_limit

    # First 429 from the burst: limit reduced once.
    manager.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    limit_after_first_429 = state.current_limit
    assert limit_after_first_429 < limit_before
    assert state.consecutive_429s == 1
    assert state.in_flight == 3

    # Second response from the same burst: 500. With the regression, this
    # would reset the cascade to 0; with the fix, in_flight > 0 keeps it at 1.
    manager.release_failure(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    assert state.consecutive_429s == 1, "cascade must not reset while the prior burst is still in-flight"
    assert state.in_flight == 2

    # Third response from the same burst: another 429. With the regression
    # this would be treated as a new cascade and reduce the limit again.
    manager.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    assert state.current_limit == limit_after_first_429, "limit must not double-reduce within the same burst"
    assert state.in_flight == 1


def test_failure_resets_cascade_after_burst_drains(manager: ThrottleManager) -> None:
    """Once the burst fully drains (in_flight == 0), the next non-RL failure breaks the cascade.

    This preserves the original PR intent for the sequential 429 → 500 → 429
    case: provider rate-limited, settled, then rate-limited again.
    """
    # Saturate, then drain: one 429 then one 500 with no concurrency.
    manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    manager.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    state = manager.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.consecutive_429s == 1
    assert state.in_flight == 0

    # New request after the burst drained. release_failure sees in_flight 1 → 0.
    manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    manager.release_failure(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    assert state.consecutive_429s == 0
    assert state.in_flight == 0


# --- Global cap ---


def test_two_aliases_effective_max_is_minimum() -> None:
    tm = ThrottleManager()
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=10)
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a2", max_parallel_requests=3)
    assert tm.get_effective_max(PROVIDER, MODEL) == 3


def test_domain_clamped_when_new_alias_lowers_cap() -> None:
    tm = ThrottleManager()
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=10)
    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.current_limit == 10

    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a2", max_parallel_requests=3)
    assert state.current_limit == 3


# --- Domain isolation ---


def test_chat_and_embedding_throttle_independently() -> None:
    tm = ThrottleManager()
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=2)

    for _ in range(2):
        tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=ThrottleDomain.CHAT, now=0.0)
    wait_chat = tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=ThrottleDomain.CHAT, now=0.0)
    assert wait_chat > 0.0

    wait_emb = tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=ThrottleDomain.EMBEDDING, now=0.0)
    assert wait_emb == 0.0


# --- 429 lifecycle scenario ---


def test_rate_limit_lifecycle_acquire_backoff_recover() -> None:
    """End-to-end AIMD lifecycle: steady-state → 429 → backoff → cooldown → recovery.

    Uses the ``now`` parameter to simulate time without real sleeps.
    Config: success_window=3, additive_increase=1, max_parallel=4, reduce_factor=0.75.
    """
    tm = ThrottleManager(ThrottleConfig(success_window=3, additive_increase=1))
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=4)
    t = 0.0

    # Phase 1 — Steady state (t=0): all 4 slots acquired and released successfully.
    for _ in range(4):
        assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=t) == 0.0
    for _ in range(4):
        tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=t)

    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.current_limit == 4

    # Phase 2 — 429 hits (t=10): reduce_factor=0.75 → floor(4*0.75)=3.
    # Domain is blocked until t=10+5=15.
    t = 10.0
    assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=t) == 0.0
    tm.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, retry_after=5.0, now=t)
    assert state.current_limit == 3
    assert state.blocked_until == 15.0

    # Phase 3 — During cooldown (t=12): acquire returns positive wait since 12 < 15.
    wait = tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=12.0)
    assert wait > 0.0

    # Phase 4 — Cooldown expires (t=16): acquire succeeds, start accumulating successes.
    # Need 3 successes (success_window=3) to bump limit 3 → 4.
    t = 16.0
    for _ in range(3):
        assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=t) == 0.0
        tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=t)
        t += 1.0

    assert state.current_limit == 4


# --- Ceiling stabilization ---


def test_ceiling_stabilization_with_overshoot() -> None:
    """After a 429, AIMD increase stops at ceiling + overshoot instead of effective_max.

    Config: effective_max=1000, success_window=1, ceiling_overshoot=0.10.
    Scenario: 429 at limit 40 → floor(40*0.75)=30 → ceiling=40 → soft cap = 40 + 4 = 44.
    Recovery should stop at 44, not climb to 1000.
    """
    tm = ThrottleManager(ThrottleConfig(success_window=1, additive_increase=1))
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=1000)
    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)

    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    state.current_limit = 40

    # 429 at limit 40 → floor(40*0.75)=30, ceiling recorded as 40.
    tm.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=10.0)
    assert state.current_limit == 30
    assert state.rate_limit_ceiling == 40

    # Pump success windows to climb back up.  soft_cap = 40 + floor(40*0.1) = 44.
    t = 20.0
    for _ in range(20):
        t += 1.0
        assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=t) == 0.0
        tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=t)

    assert state.current_limit == 44, f"Expected stabilization at 44, got {state.current_limit}"

    # Further successes should not increase beyond the soft ceiling.
    for _ in range(10):
        t += 1.0
        assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=t) == 0.0
        tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=t)

    assert state.current_limit == 44, f"Limit crept past soft ceiling: {state.current_limit}"


def test_ceiling_lowers_on_repeated_429_after_recovery() -> None:
    """A 429 after partial recovery lowers the ceiling, tightening the soft cap.

    Scenario: first 429 at 40 → floor(40*0.75)=30, ceiling=40.
    Recovery: set limit to 30, one success bumps to 31 (success_window=1).
    Second 429 at 31 → floor(31*0.75)=23, ceiling = min(40, 31) = 31.
    Soft cap = 31 + max(1, floor(31*0.1)) = 31 + 3 = 34.
    """
    tm = ThrottleManager(ThrottleConfig(success_window=1, additive_increase=1))
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=1000)

    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    state.current_limit = 40

    # First 429 at 40 → floor(40*0.75)=30, ceiling=40.
    tm.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=10.0)
    assert state.rate_limit_ceiling == 40
    assert state.current_limit == 30

    # Recovery: one success bumps 30 → 31.
    t = 20.0
    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=t)
    tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=t)
    assert state.current_limit == 31

    # Second 429 at 31 → floor(31*0.75)=23, ceiling = min(40, 31) = 31.
    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=t + 1)
    tm.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=t + 1)
    assert state.rate_limit_ceiling == 31
    assert state.current_limit == 23

    # Soft cap = 31 + max(1, floor(31*0.1)) = 34. Climb should stop there.
    t = 40.0
    for _ in range(15):
        t += 1.0
        assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=t) == 0.0
        tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=t)

    assert state.current_limit == 34, f"Expected soft cap at 34, got {state.current_limit}"


def test_cascade_only_first_429_reduces_limit() -> None:
    """Only the first 429 in a cascade reduces the limit; subsequent ones just release permits."""
    tm = ThrottleManager(ThrottleConfig(success_window=1, additive_increase=1))
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=100)

    for _ in range(4):
        tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)

    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.in_flight == 4

    # First 429: limit 100 → 75, ceiling set to 100.
    tm.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=1.0)
    assert state.current_limit == 75
    assert state.rate_limit_ceiling == 100
    assert state.in_flight == 3

    # Subsequent cascade 429s: limit stays at 75, only in_flight decrements.
    tm.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=1.0)
    assert state.current_limit == 75
    assert state.rate_limit_ceiling == 100
    assert state.in_flight == 2

    tm.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=1.0)
    assert state.current_limit == 75
    assert state.in_flight == 1

    tm.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=1.0)
    assert state.current_limit == 75
    assert state.in_flight == 0


def test_ceiling_does_not_restrict_when_at_effective_max() -> None:
    """When effective_max is small (e.g. 4), the ceiling + overshoot should not
    prevent recovery to effective_max.
    """
    tm = ThrottleManager(ThrottleConfig(success_window=1, additive_increase=1))
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=4)

    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    tm.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=10.0)

    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    # floor(4 * 0.75) = 3; ceiling=4, soft_cap = min(4 + max(1, floor(4*0.1)), 4) = 4
    assert state.current_limit == 3

    t = 20.0
    for _ in range(5):
        t += 1.0
        assert tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=t) == 0.0
        tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=t)

    assert state.current_limit == 4, f"Should recover to effective_max=4, got {state.current_limit}"


# --- Acquire timeout ---


def test_acquire_sync_raises_timeout_when_at_capacity() -> None:
    tm = ThrottleManager()
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=1)
    # Saturate the single slot so try_acquire returns a positive wait.
    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN)

    with pytest.raises(TimeoutError, match="timed out"):
        tm.acquire_sync(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, timeout=0.0)


def test_acquire_sync_does_not_overshoot_timeout() -> None:
    """When wait > remaining budget, raise immediately instead of sleeping the full wait."""
    tm = ThrottleManager(ThrottleConfig(cooldown_seconds=5.0))
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=1)
    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN)

    # Timeout of 0.5s is less than the 5s block wait — should raise fast, not sleep 5s.
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out"):
        tm.acquire_sync(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, timeout=0.5)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, f"acquire_sync overshot timeout: elapsed {elapsed:.1f}s (expected <2s)"


@pytest.mark.asyncio
async def test_acquire_async_raises_timeout_when_at_capacity() -> None:
    tm = ThrottleManager()
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=1)
    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN)

    with pytest.raises(TimeoutError, match="timed out"):
        await tm.acquire_async(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, timeout=0.0)


@pytest.mark.asyncio
async def test_acquire_async_default_no_deadline_waits_for_release() -> None:
    """``timeout=None`` (the default) waits for the permit instead of raising.

    Issue #551: the previous 300s default produced spurious ``ModelTimeoutError``
    cascades on slow endpoints with deep queues; now queue waits scale with
    provider speed and only the HTTP timeout deadlines actual work. The
    ``timeout=0.0`` case is covered by ``test_acquire_async_raises_timeout_when_at_capacity``.
    """
    tm = ThrottleManager()
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=1)
    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN)

    async def release_after(delay: float) -> None:
        await asyncio.sleep(delay)
        tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN)

    # Hold a strong reference to the task so the loop's weak-ref bookkeeping
    # can't GC it before the inner await observes the release.
    release_task = asyncio.create_task(release_after(0.05))
    try:
        # asyncio.wait_for caps the test runtime; the inner acquire_async passes None.
        await asyncio.wait_for(
            tm.acquire_async(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN),
            timeout=2.0,
        )
    finally:
        await release_task


def test_acquire_sync_default_no_deadline_waits_for_release() -> None:
    """Sync counterpart: ``timeout=None`` default blocks until release."""
    tm = ThrottleManager()
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=1)
    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN)

    def release_after(delay: float) -> None:
        time.sleep(delay)
        tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN)

    threading.Thread(target=release_after, args=(0.05,), daemon=True).start()
    start = time.monotonic()
    tm.acquire_sync(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN)
    elapsed = time.monotonic() - start
    assert 0.04 < elapsed < 2.0, f"expected ~0.05s wait, got {elapsed:.3f}s"


# --- Thread safety ---


def test_concurrent_acquire_release_no_errors() -> None:
    tm = ThrottleManager()
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=4)
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for _ in range(50):
                tm.acquire_sync(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN)
                tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors, f"Thread errors: {errors}"

    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.in_flight == 0
