# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-wide event-loop management for async engine work.

Singleton event loop:
    The background loop is a process-wide singleton. Async-stateful
    resources (connection pools, semaphores) bind internal state to a
    specific event loop, so creating per-call or per-instance loops breaks
    connection reuse and triggers cross-loop errors.
    ``ensure_async_engine_loop()`` creates one daemon loop thread and reuses
    it for all async engine work.

Startup handshake:
    Loop creation uses a ``threading.Event`` readiness handshake. The
    background thread signals readiness via ``loop.call_soon(ready.set)``,
    and the creating thread holds the lock until that event fires (or a
    timeout expires). This prevents a race where a second caller could see
    ``_loop.is_running() == False`` before the first loop has entered
    ``run_forever()``, which would create a duplicate loop. On timeout,
    globals are reset and the orphaned loop is cleaned up before raising.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from data_designer.config.run_config import RunConfig

logger = logging.getLogger(__name__)

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()

_LOOP_READY_TIMEOUT = 5.0  # seconds to wait for the background loop to start


class _CancellableScheduler(Protocol):
    def request_cancel(self) -> None: ...


def is_async_trace_enabled(settings: RunConfig) -> bool:
    return settings.async_trace or os.environ.get("DATA_DESIGNER_ASYNC_TRACE", "0") == "1"


def await_async_scheduler_result(
    future: concurrent.futures.Future[Any],
    scheduler: _CancellableScheduler,
) -> None:
    """Wait for a scheduler result and let cancellation cleanup finish after an interrupt."""
    try:
        future.result()
    except KeyboardInterrupt:
        scheduler.request_cancel()
        try:
            future.result()
        except concurrent.futures.CancelledError:
            pass
        except Exception:
            logger.debug("Async scheduler raised while cancelling after KeyboardInterrupt", exc_info=True)
        raise


def _run_loop(loop: asyncio.AbstractEventLoop, ready: threading.Event) -> None:
    asyncio.set_event_loop(loop)
    loop.call_soon(ready.set)
    loop.run_forever()


def ensure_async_engine_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop for async engine work.

    A single event loop is shared across async engine work to avoid breaking
    async-stateful resources that bind internal state to a specific event loop.
    """
    global _loop, _thread
    with _lock:
        if _loop is None or not _loop.is_running():
            ready = threading.Event()
            _loop = asyncio.new_event_loop()
            _thread = threading.Thread(target=_run_loop, args=(_loop, ready), daemon=True, name="AsyncEngine-EventLoop")
            _thread.start()
            if not ready.wait(timeout=_LOOP_READY_TIMEOUT):
                orphan_loop = _loop
                orphan_thread = _thread
                _loop = None
                _thread = None

                if orphan_loop is not None:
                    try:
                        if orphan_thread is not None and orphan_thread.is_alive():
                            orphan_loop.call_soon_threadsafe(orphan_loop.stop)
                        if not orphan_loop.is_running():
                            orphan_loop.close()
                    except Exception:
                        logger.warning("Failed to clean up timed-out AsyncEngine loop startup", exc_info=True)

                raise RuntimeError("AsyncEngine event loop failed to start within timeout")
    return _loop
