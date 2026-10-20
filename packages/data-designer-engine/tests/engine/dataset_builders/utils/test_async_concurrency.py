# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import multiprocessing

import pytest

from data_designer.engine.dataset_builders.utils import async_concurrency

_CHILD_JOIN_TIMEOUT_S = 20.0

pytestmark = [
    pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires the fork start method"),
    pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning"),
]


async def _ping() -> str:
    return "pong"


def _ping_engine_loop(queue: multiprocessing.Queue) -> None:
    loop = async_concurrency.ensure_async_engine_loop()
    queue.put(asyncio.run_coroutine_threadsafe(_ping(), loop).result(timeout=5))


def _run_forked_child() -> str:
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    child = ctx.Process(target=_ping_engine_loop, args=(queue,))
    child.start()
    child.join(timeout=_CHILD_JOIN_TIMEOUT_S)
    if child.is_alive():
        child.kill()
        child.join()
        pytest.fail("forked child blocked on the AsyncEngine event loop inherited from the parent")
    assert child.exitcode == 0, "forked child could not use the AsyncEngine event loop"
    return queue.get(timeout=5)


def test_forked_child_starts_fresh_loop_after_parent_used_it() -> None:
    parent_loop = async_concurrency.ensure_async_engine_loop()
    assert asyncio.run_coroutine_threadsafe(_ping(), parent_loop).result(timeout=5) == "pong"

    assert _run_forked_child() == "pong"

    assert async_concurrency.ensure_async_engine_loop() is parent_loop
    assert asyncio.run_coroutine_threadsafe(_ping(), parent_loop).result(timeout=5) == "pong"


def test_forked_child_does_not_inherit_held_loop_lock() -> None:
    with async_concurrency._lock:
        result = _run_forked_child()
    assert result == "pong"
