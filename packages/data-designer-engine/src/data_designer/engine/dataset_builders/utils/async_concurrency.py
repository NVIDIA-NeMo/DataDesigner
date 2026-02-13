# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async batch execution with bounded concurrency and early-shutdown semantics.

Async counterpart to ``concurrency.py``. Same operational contract (callbacks
with optional context, error aggregation, shutdown thresholds), different
runtime model. The sync module runs callables in a ``ThreadPoolExecutor``;
this module runs coroutines in ``asyncio.TaskGroup`` on a dedicated loop
thread. Callers stay synchronous.

Architecture:
    ``AsyncConcurrentExecutor.run()`` is a blocking call that submits
    coroutines to a shared background event loop via
    ``run_coroutine_threadsafe``. Bounded concurrency is enforced with an
    ``asyncio.Semaphore``. Success/error counts use the same
    ``ExecutorResults`` model as the sync executor.

    Caller Thread ──► run() ──► run_coroutine_threadsafe ──► Background Loop
                                                              (TaskGroup)

Singleton Event Loop:
    The background loop is a process-wide singleton. LiteLLM and similar
    libraries bind internal async state to a specific event loop, so creating
    per-call or per-instance loops breaks connection reuse and triggers
    cross-loop errors. ``_ensure_async_engine_loop()`` creates one daemon
    loop thread and reuses it for all executor instances.

Startup Handshake:
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
import json
import logging
import threading
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from data_designer.engine.dataset_builders.utils.concurrency import (
    CallbackWithContext,
    ErrorCallbackWithContext,
    ExecutorResults,
)
from data_designer.engine.errors import DataDesignerRuntimeError
from data_designer.logging import LOG_INDENT

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Success(Generic[T]):
    index: int
    value: T


@dataclass(frozen=True, slots=True)
class Failure:
    index: int
    error: Exception


TaskResult = Success[T] | Failure

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()

_LOOP_READY_TIMEOUT = 5.0  # seconds to wait for the background loop to start


def _run_loop(loop: asyncio.AbstractEventLoop, ready: threading.Event) -> None:
    asyncio.set_event_loop(loop)
    loop.call_soon(ready.set)
    loop.run_forever()


def _ensure_async_engine_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop for async engine work.

    A single event loop is shared across all AsyncConcurrentExecutor instances
    to avoid breaking libraries (like LiteLLM) that bind internal async state
    to a specific event loop.
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


class AsyncConcurrentExecutor:
    """Async equivalent of ConcurrentThreadExecutor.

    Executes a batch of coroutines with bounded concurrency, error rate
    monitoring, and early shutdown semantics. Callers remain synchronous —
    the ``run()`` method submits work to a persistent background event loop.

    No locks are needed because asyncio tasks run cooperatively on a
    single thread — mutations to ``_results`` are always sequential.
    """

    def __init__(
        self,
        *,
        max_workers: int,
        column_name: str,
        result_callback: CallbackWithContext | None = None,
        error_callback: ErrorCallbackWithContext | None = None,
        shutdown_error_rate: float = 0.50,
        shutdown_error_window: int = 10,
        disable_early_shutdown: bool = False,
    ) -> None:
        self._column_name = column_name
        self._max_workers = max_workers
        self._result_callback = result_callback
        self._error_callback = error_callback
        self._shutdown_error_rate = shutdown_error_rate
        self._shutdown_window_size = shutdown_error_window
        self._disable_early_shutdown = disable_early_shutdown
        self._results = ExecutorResults(failure_threshold=shutdown_error_rate)

    @property
    def results(self) -> ExecutorResults:
        return self._results

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @property
    def shutdown_error_rate(self) -> float:
        return self._shutdown_error_rate

    @property
    def shutdown_window_size(self) -> int:
        return self._shutdown_window_size

    def run(self, work_items: list[tuple[Coroutine[Any, Any, Any], dict | None]]) -> None:
        """Execute all work items concurrently. Callers remain synchronous."""
        logger.debug(
            f"AsyncConcurrentExecutor: launching {len(work_items)} tasks "
            f"with max_workers={self._max_workers} for column '{self._column_name}'"
        )
        loop = _ensure_async_engine_loop()
        future = asyncio.run_coroutine_threadsafe(self._run_all(work_items), loop)
        future.result()

    async def _run_all(self, work_items: list[tuple[Coroutine[Any, Any, Any], dict | None]]) -> None:
        self._semaphore = asyncio.Semaphore(self._max_workers)
        self._shutdown_event = asyncio.Event()

        async with asyncio.TaskGroup() as tg:
            for i, (coro, context) in enumerate(work_items):
                tg.create_task(self._run_task(i, coro, context))

        if not self._disable_early_shutdown and self._results.early_shutdown:
            self._raise_task_error()

    async def _run_task(self, index: int, coro: Coroutine[Any, Any, Any], context: dict | None) -> None:
        if self._shutdown_event.is_set():
            coro.close()
            return

        async with self._semaphore:
            if self._shutdown_event.is_set():
                coro.close()
                return

            try:
                result = await coro
                if self._result_callback is not None:
                    self._result_callback(result, context=context)
                self._results.completed_count += 1
                self._results.success_count += 1
            except Exception as err:
                self._results.completed_count += 1
                self._results.error_trap.handle_error(err)
                if not self._disable_early_shutdown and self._results.is_error_rate_exceeded(
                    self._shutdown_window_size
                ):
                    if not self._results.early_shutdown:
                        self._results.early_shutdown = True
                    self._shutdown_event.set()
                if self._error_callback is not None:
                    try:
                        self._error_callback(err, context=context)
                    except Exception:
                        logger.warning("error_callback raised an exception", exc_info=True)

    def _raise_task_error(self) -> None:
        raise DataDesignerRuntimeError(
            "\n".join(
                [
                    f"{LOG_INDENT}Data generation was terminated early due to error rate exceeding threshold.",
                    f"{LOG_INDENT}The summary of encountered errors is: \n{json.dumps(self._results.summary, indent=4)}",
                ]
            )
        )
