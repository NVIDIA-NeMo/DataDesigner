# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextvars
import faulthandler
import json
import logging
from concurrent.futures import ALL_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from threading import Lock, Semaphore
from typing import Any, Protocol

from pydantic import BaseModel, Field

from data_designer.engine.errors import DataDesignerRuntimeError, ErrorTrap

logger = logging.getLogger(__name__)

# Constants
MAX_CONCURRENCY_PER_NON_LLM_GENERATOR = 4


class ExecutorResults(BaseModel):
    failure_threshold: float = 0.0  # Error rate threshold
    submitted_count: int = 0  # How many tasks/jobs were submitted
    completed_count: int = 0  # How many tasks/jobs completed
    success_count: int = 0  # How many tasks/jobs were successful
    early_shutdown: bool = False  # Did we shutdown early due to errors?
    error_trap: ErrorTrap = Field(default_factory=ErrorTrap)
    last_completed_at_utc: str | None = None  # ISO timestamp of most recent completion

    @property
    def summary(self) -> dict:
        summary = self.model_dump(exclude={"error_trap"})
        summary |= self.error_trap.model_dump()
        return summary

    def get_error_rate(self, window: int) -> float:
        # We don't start actually tracking until our minimum window size is met
        if self.completed_count < window:
            return 0.0
        return self.error_trap.error_count / max(1, self.completed_count)

    def is_error_rate_exceeded(self, window: int) -> bool:
        return self.get_error_rate(window) >= self.failure_threshold


class CallbackWithContext(Protocol):
    """Executor callback functions must accept a context kw argument."""

    def __call__(self, result: Any, *, context: dict | None = None) -> Any: ...


class ErrorCallbackWithContext(Protocol):
    """Error callbacks take the Exception instance and context."""

    def __call__(self, exc: Exception, *, context: dict | None = None) -> Any: ...


class ConcurrentThreadExecutor:
    """
    Interface for executing multiple concurrent tasks with error rate monitoring.

    This interface should be used exclusively as
    a context manager. New tasks can be submitted to the executor using the `submit`
    method. This submit method functions similarly to the
    submit method of a ThreadPoolExecutor.

    The underlying queue of tasks is bounded by the `max_workers`
    parameter. This means that only `max_workers` number of
    tasks can be queued up for execution. As tasks complete,
    if there are errors, those are tracked and counted. If
    a certain error rate is exceeded, the executor will shutdown
    early. All queued and running tasks will complete.

    The reason we bound the underlying task queue is to ensure that when
    a certain error threshold is met there aren't an unbounded
    number of tasks that need to complete. Generally speaking,
    tasks should not be sitting in the queue for long at all since
    the queue size == `max_workers`. The side effect of this is that
    the `submit()` method will block, however this should not matter
    because upstream Tasks need to wait for all jobs to complete
    before the Task can be considered complete.

    ContextVars from the main parent thread are automatically propagated
    to all child threads.

    When a task is completed, the user provided `result_callback`
    function will be called with the task result as the only argument.
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
    ):
        self._executor = None
        self._column_name = column_name
        self._max_workers = max_workers
        self._lock = Lock()
        self._semaphore = Semaphore(self._max_workers)
        self._result_callback = result_callback
        self._error_callback = error_callback
        self._shutdown_error_rate = shutdown_error_rate
        self._shutdown_window_size = shutdown_error_window
        self._disable_early_shutdown = disable_early_shutdown
        self._results = ExecutorResults(failure_threshold=shutdown_error_rate)
        self._futures: list[Future] = []
        self._future_context: dict[Future, dict | None] = {}
        self._shutdown_wait: bool = True

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

    @property
    def semaphore(self) -> Semaphore:
        return self._semaphore

    def __enter__(self) -> ConcurrentThreadExecutor:
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="ConcurrentThreadExecutor",
            initializer=_set_worker_contextvars,
            initargs=(contextvars.copy_context(),),
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # If we are exiting because of an exception or a detected hang, don't block forever waiting
        # on worker threads that may be stuck (e.g. IO deadlocks).
        wait_for_workers = exc_type is None and self._shutdown_wait
        self._shutdown_executor(wait=wait_for_workers, cancel_futures=not wait_for_workers)
        if not self._disable_early_shutdown and self._results.early_shutdown is True:
            self._raise_task_error()

    def _shutdown_executor(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def _raise_task_error(self):
        raise DataDesignerRuntimeError(
            "\n".join(
                [
                    "  |-- Data generation was terminated early due to error rate exceeding threshold.",
                    f"  |-- The summary of encountered errors is: \n{json.dumps(self._results.summary, indent=4)}",
                ]
            )
        )

    def submit(
        self,
        fn,
        *args,
        context: dict | None = None,
        acquire_timeout_s: float | None = None,
        dump_stacks_on_timeout: bool = False,
        **kwargs,
    ) -> None:
        if self._executor is None:
            raise RuntimeError("Executor is not initialized, this class should be used as a context manager.")

        if not self._disable_early_shutdown and self._results.early_shutdown:
            self._shutdown_executor()
            self._raise_task_error()

        def _handle_future(future: Future) -> None:
            try:
                result = future.result()
                if self._result_callback is not None:
                    self._result_callback(result, context=context)
                with self._lock:
                    self._results.completed_count += 1
                    self._results.success_count += 1
                    self._results.last_completed_at_utc = datetime.now(timezone.utc).isoformat()
            except Exception as err:
                with self._lock:
                    self._results.completed_count += 1
                    self._results.last_completed_at_utc = datetime.now(timezone.utc).isoformat()
                    self._results.error_trap.handle_error(err)
                    if not self._disable_early_shutdown and self._results.is_error_rate_exceeded(
                        self._shutdown_window_size
                    ):
                        # Signal to shutdown early on the next submission (if received).
                        # We cannot trigger shutdown from within this thread as it can
                        # cause a deadlock.
                        if not self._results.early_shutdown:
                            self._results.early_shutdown = True
                if self._error_callback is not None:
                    self._error_callback(err, context=context)
            finally:
                with self._lock:
                    self._future_context.pop(future, None)
                self._semaphore.release()

        acquired = False
        try:
            acquired = (
                self._semaphore.acquire(timeout=acquire_timeout_s)
                if acquire_timeout_s is not None
                else self._semaphore.acquire()
            )
            if not acquired:
                self._shutdown_wait = False
                if dump_stacks_on_timeout:
                    faulthandler.dump_traceback(all_threads=True)
                raise DataDesignerRuntimeError(
                    "\n".join(
                        [
                            f"🛑 Timed out acquiring executor capacity for column {self._column_name!r}.",
                            f"  |-- max_workers={self._max_workers}",
                            f"  |-- submitted={self._results.submitted_count}",
                            f"  |-- completed={self._results.completed_count}",
                            f"  |-- last_completed_at_utc={self._results.last_completed_at_utc}",
                            f"  |-- blocked_context={context}",
                        ]
                    )
                )

            future = self._executor.submit(fn, *args, **kwargs)
            with self._lock:
                self._futures.append(future)
                self._future_context[future] = context
                self._results.submitted_count += 1
            future.add_done_callback(_handle_future)
        except Exception as err:
            # If we get here, the pool is shutting down (likely due to early termination from errors)
            # We'll re-raise a custom error that can be handled at the call-site and the summary
            # can also be inspected.
            if acquired:
                self._semaphore.release()
            is_shutdown_error = isinstance(err, RuntimeError) and (
                "after shutdown" in str(err) or "Pool shutdown" in str(err)
            )
            if not is_shutdown_error:
                raise err
            if self._disable_early_shutdown:
                raise err
            self._raise_task_error()

    def wait_for_completion(self, *, timeout_s: float | None, dump_stacks_on_timeout: bool = False) -> None:
        """Wait for all submitted work to complete, with optional timeout and diagnostics."""
        if self._executor is None:
            raise RuntimeError("Executor is not initialized, this class should be used as a context manager.")

        with self._lock:
            futures_snapshot = list(self._futures)

        if not futures_snapshot:
            return

        done, not_done = wait(futures_snapshot, timeout=timeout_s, return_when=ALL_COMPLETED)
        if not not_done:
            return

        self._shutdown_wait = False
        if dump_stacks_on_timeout:
            faulthandler.dump_traceback(all_threads=True)

        with self._lock:
            pending_contexts = [self._future_context.get(fut) for fut in not_done]
        pending_contexts = [ctx for ctx in pending_contexts if ctx is not None]
        pending_contexts_sample = pending_contexts[:20]

        # Best effort cancellation of any queued-but-not-started tasks; running tasks cannot be cancelled.
        self._shutdown_executor(wait=False, cancel_futures=True)

        raise DataDesignerRuntimeError(
            "\n".join(
                [
                    f"🛑 Timed out waiting for column {self._column_name!r} to finish parallel work.",
                    f"  |-- max_workers={self._max_workers}",
                    f"  |-- submitted={self._results.submitted_count}",
                    f"  |-- completed={self._results.completed_count}",
                    f"  |-- unfinished={len(not_done)}",
                    f"  |-- last_completed_at_utc={self._results.last_completed_at_utc}",
                    f"  |-- pending_contexts_sample={pending_contexts_sample}",
                ]
            )
        )


def _set_worker_contextvars(context: contextvars.Context):
    for var, value in context.items():
        var.set(value)
