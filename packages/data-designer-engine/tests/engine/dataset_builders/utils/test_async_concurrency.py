# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from tests.engine.fork_probe import requires_fork, run_fork_probe


@requires_fork
def test_forked_child_starts_fresh_loop_after_parent_used_it() -> None:
    run_fork_probe(
        """
        from data_designer.engine.dataset_builders.utils import async_concurrency


        def child(queue):
            loop = async_concurrency.ensure_async_engine_loop()
            queue.put(asyncio.run_coroutine_threadsafe(ping(), loop).result(timeout=5))


        parent_loop = async_concurrency.ensure_async_engine_loop()
        assert asyncio.run_coroutine_threadsafe(ping(), parent_loop).result(timeout=5) == "pong"
        assert run_forked_child(child) == "pong"
        assert async_concurrency.ensure_async_engine_loop() is parent_loop
        assert asyncio.run_coroutine_threadsafe(ping(), parent_loop).result(timeout=5) == "pong"
        """
    )


@requires_fork
def test_forked_child_does_not_inherit_held_loop_lock() -> None:
    run_fork_probe(
        """
        from data_designer.engine.dataset_builders.utils import async_concurrency


        def child(queue):
            loop = async_concurrency.ensure_async_engine_loop()
            queue.put(asyncio.run_coroutine_threadsafe(ping(), loop).result(timeout=5))


        with async_concurrency._lock:
            assert run_forked_child(child) == "pong"
        """
    )
