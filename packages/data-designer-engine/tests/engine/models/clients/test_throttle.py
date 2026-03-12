# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

import pytest

from data_designer.engine.models.clients.throttle import (
    DEFAULT_BLOCK_SECONDS,
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


def test_acquire_at_capacity_returns_positive_wait(manager: ThrottleManager) -> None:
    for _ in range(4):
        manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    wait = manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    assert wait > 0.0


def test_acquire_respects_blocked_until(manager: ThrottleManager) -> None:
    manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    manager.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, retry_after=5.0, now=1.0)
    wait = manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=2.0)
    assert wait == pytest.approx(4.0, abs=0.01)


def test_acquire_without_registration_uses_min_limit() -> None:
    tm = ThrottleManager()
    assert tm.try_acquire(provider_name="unknown", model_id="m", domain=DOMAIN, now=0.0) == 0.0
    assert tm.try_acquire(provider_name="unknown", model_id="m", domain=DOMAIN, now=0.0) > 0.0


# --- release_success ---


def test_release_success_frees_slot(manager: ThrottleManager) -> None:
    for _ in range(4):
        manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    manager.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    wait = manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    assert wait == 0.0


def test_additive_increase_after_success_window() -> None:
    tm = ThrottleManager(success_window=5)
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
    tm = ThrottleManager(success_window=1, additive_increase=3)
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
    tm = ThrottleManager(success_window=1)
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=2)
    for i in range(20):
        tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=float(i))
        tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=float(i))
    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.current_limit <= 2


def test_additive_increase_clamped_to_effective_max() -> None:
    tm = ThrottleManager(success_window=1, additive_increase=100)
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=5)
    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    tm.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)

    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=1.0)
    tm.release_success(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=1.0)

    state = tm.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.current_limit == 5


# --- release_rate_limited ---


def test_rate_limited_halves_current_limit(manager: ThrottleManager) -> None:
    manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    manager.release_rate_limited(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    state = manager.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    assert state.current_limit == 2


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
    assert state.blocked_until == pytest.approx(10.0 + DEFAULT_BLOCK_SECONDS, abs=0.01)


# --- release_failure ---


def test_failure_releases_slot_without_limit_change(manager: ThrottleManager) -> None:
    manager.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    state = manager.get_domain_state(PROVIDER, MODEL, DOMAIN)
    assert state is not None
    limit_before = state.current_limit
    manager.release_failure(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, now=0.0)
    assert state.current_limit == limit_before
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


# --- Acquire timeout ---


def test_acquire_sync_raises_timeout_when_at_capacity() -> None:
    tm = ThrottleManager()
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=1)
    # Saturate the single slot so try_acquire returns a positive wait.
    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN)

    with pytest.raises(TimeoutError, match="timed out"):
        tm.acquire_sync(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, timeout=0.0)


@pytest.mark.asyncio
async def test_acquire_async_raises_timeout_when_at_capacity() -> None:
    tm = ThrottleManager()
    tm.register(provider_name=PROVIDER, model_id=MODEL, alias="a1", max_parallel_requests=1)
    tm.try_acquire(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN)

    with pytest.raises(TimeoutError, match="timed out"):
        await tm.acquire_async(provider_name=PROVIDER, model_id=MODEL, domain=DOMAIN, timeout=0.0)


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
