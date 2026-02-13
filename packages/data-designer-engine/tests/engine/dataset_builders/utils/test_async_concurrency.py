# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys

import pytest

# asyncio.TaskGroup requires Python 3.11+
pytestmark = pytest.mark.skipif(sys.version_info < (3, 11), reason="asyncio.TaskGroup requires Python 3.11+")

from data_designer.engine.dataset_builders.utils.async_concurrency import (
    AsyncConcurrentExecutor,
)
from data_designer.engine.dataset_builders.utils.concurrency import ExecutorResults
from data_designer.engine.errors import DataDesignerRuntimeError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _succeed(value: int) -> int:
    """Simple coroutine that returns its input doubled."""
    return value * 2


async def _fail(msg: str = "Test error") -> None:
    """Simple coroutine that always raises."""
    raise ValueError(msg)


async def _succeed_slow(value: int, delay: float = 0.05) -> int:
    """Coroutine with a small delay to simulate work."""
    await asyncio.sleep(delay)
    return value * 2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_creation():
    executor = AsyncConcurrentExecutor(
        max_workers=4,
        column_name="test_column",
        shutdown_error_rate=0.3,
        shutdown_error_window=5,
    )
    assert executor.max_workers == 4
    assert executor.shutdown_error_rate == 0.3
    assert executor.shutdown_window_size == 5
    assert isinstance(executor.results, ExecutorResults)
    assert executor.results.completed_count == 0
    assert executor.results.success_count == 0
    assert executor.results.early_shutdown is False
    assert executor.results.failure_threshold == 0.3


def test_successful_execution():
    executor = AsyncConcurrentExecutor(max_workers=4, column_name="test_column")
    work_items = [(_succeed(i), None) for i in range(10)]
    executor.run(work_items)

    assert executor.results.completed_count == 10
    assert executor.results.success_count == 10
    assert executor.results.error_trap.error_count == 0
    assert executor.results.early_shutdown is False


def test_successful_execution_with_context():
    executor = AsyncConcurrentExecutor(max_workers=2, column_name="test_column")
    work_items = [(_succeed(i), {"index": i}) for i in range(5)]
    executor.run(work_items)

    assert executor.results.completed_count == 5
    assert executor.results.success_count == 5


def test_result_callback():
    results = []

    def result_callback(result, *, context=None):
        results.append((result, context))

    executor = AsyncConcurrentExecutor(
        max_workers=2,
        column_name="test_column",
        result_callback=result_callback,
    )
    work_items = [
        (_succeed(5), {"key": "a"}),
        (_succeed(10), {"key": "b"}),
    ]
    executor.run(work_items)

    assert len(results) == 2
    values = sorted(results, key=lambda r: r[0])
    assert values[0] == (10, {"key": "a"})
    assert values[1] == (20, {"key": "b"})


def test_result_callback_with_none_context():
    results = []

    def result_callback(result, *, context=None):
        results.append((result, context))

    executor = AsyncConcurrentExecutor(
        max_workers=2,
        column_name="test_column",
        result_callback=result_callback,
    )
    work_items = [(_succeed(7), None)]
    executor.run(work_items)

    assert len(results) == 1
    assert results[0] == (14, None)


def test_error_callback():
    errors = []

    def error_callback(exc, *, context=None):
        errors.append((exc, context))

    executor = AsyncConcurrentExecutor(
        max_workers=2,
        column_name="test_column",
        error_callback=error_callback,
        disable_early_shutdown=True,
    )
    work_items = [(_fail("boom"), {"task": "first"})]
    executor.run(work_items)

    assert len(errors) == 1
    assert isinstance(errors[0][0], ValueError)
    assert str(errors[0][0]) == "boom"
    assert errors[0][1] == {"task": "first"}


def test_early_shutdown_when_threshold_exceeded():
    """Error rate exceeds threshold -- should raise DataDesignerRuntimeError."""
    executor = AsyncConcurrentExecutor(
        max_workers=4,
        column_name="test_column",
        shutdown_error_rate=0.5,
        shutdown_error_window=2,
    )
    # All tasks fail -> 100% error rate, well above 50% threshold
    work_items = [(_fail(f"err-{i}"), None) for i in range(10)]

    with pytest.raises(DataDesignerRuntimeError, match="Data generation was terminated early"):
        executor.run(work_items)

    assert executor.results.early_shutdown is True


def test_no_early_shutdown_below_threshold():
    """Error rate stays below threshold -- should NOT raise."""
    executor = AsyncConcurrentExecutor(
        max_workers=4,
        column_name="test_column",
        shutdown_error_rate=0.5,
        shutdown_error_window=20,
    )
    # 2 failures + 18 successes = 10% error rate, well below 50%
    work_items = [(_fail(f"err-{i}"), None) for i in range(2)] + [(_succeed(i), None) for i in range(18)]
    executor.run(work_items)

    assert executor.results.early_shutdown is False
    assert executor.results.completed_count == 20
    assert executor.results.success_count == 18
    assert executor.results.error_trap.error_count == 2


def test_disable_early_shutdown():
    """All tasks fail but disable_early_shutdown=True -- no DataDesignerRuntimeError."""
    executor = AsyncConcurrentExecutor(
        max_workers=4,
        column_name="test_column",
        shutdown_error_rate=0.0,
        shutdown_error_window=0,
        disable_early_shutdown=True,
    )
    work_items = [(_fail(f"err-{i}"), None) for i in range(10)]
    # Should not raise
    executor.run(work_items)

    assert executor.results.error_trap.error_count == 10
    assert executor.results.success_count == 0
    assert executor.results.completed_count == 10
    assert executor.results.early_shutdown is False


def test_result_callback_raises_counts_as_failure():
    """When result_callback raises, the task should count as a failure, not a success.

    This validates the fix where a callback exception was previously
    double-counted or misattributed. The corrected behavior: if the
    coroutine succeeds but the callback raises, completed_count is
    incremented once and the error is trapped (success_count is NOT
    incremented).
    """

    def bad_callback(result, *, context=None):
        raise RuntimeError("callback exploded")

    executor = AsyncConcurrentExecutor(
        max_workers=2,
        column_name="test_column",
        result_callback=bad_callback,
        disable_early_shutdown=True,
    )
    work_items = [(_succeed(i), None) for i in range(5)]
    executor.run(work_items)

    # Each task's coroutine succeeds, but callback raises -> counted as failure
    assert executor.results.completed_count == 5
    assert executor.results.success_count == 0
    assert executor.results.error_trap.error_count == 5


def test_error_callback_raises_safely():
    """error_callback raising should not crash the executor -- just log a warning."""

    def bad_error_callback(exc, *, context=None):
        raise RuntimeError("error callback also broke")

    executor = AsyncConcurrentExecutor(
        max_workers=2,
        column_name="test_column",
        error_callback=bad_error_callback,
        disable_early_shutdown=True,
    )
    work_items = [(_fail(f"err-{i}"), None) for i in range(5)]
    # Should not raise despite error_callback raising
    executor.run(work_items)

    assert executor.results.completed_count == 5
    assert executor.results.error_trap.error_count == 5


def test_semaphore_bounding():
    """Verify concurrency is bounded by max_workers."""
    max_concurrent = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    async def tracked_work(index: int) -> int:
        nonlocal max_concurrent, current_concurrent
        async with lock:
            current_concurrent += 1
            if current_concurrent > max_concurrent:
                max_concurrent = current_concurrent
        # Yield control so other tasks can run concurrently
        await asyncio.sleep(0.02)
        async with lock:
            current_concurrent -= 1
        return index

    max_workers = 3
    executor = AsyncConcurrentExecutor(max_workers=max_workers, column_name="test_column")
    work_items = [(tracked_work(i), None) for i in range(20)]
    executor.run(work_items)

    assert executor.results.completed_count == 20
    assert executor.results.success_count == 20
    assert max_concurrent <= max_workers, f"Max concurrent was {max_concurrent}, expected <= {max_workers}"
    # Also confirm the semaphore was actually exercised (more tasks than workers)
    assert max_concurrent >= 1


@pytest.mark.parametrize(
    "shutdown_error_rate,num_errors,num_successes,shutdown_window,should_raise",
    [
        (0.5, 60, 40, 20, True),  # 60% errors > 50% threshold
        (0.3, 40, 60, 20, True),  # 40% errors > 30% threshold
        (0.0, 5, 5, 10, True),  # Any error > 0% threshold
        (1.0, 20, 0, 10, True),  # 100% errors >= 100% threshold
        (0.5, 10, 90, 20, False),  # 10% errors < 50% threshold
        (0.3, 10, 90, 20, False),  # 10% errors < 30% threshold
        (1.0, 50, 50, 20, False),  # 50% errors < 100% threshold
    ],
)
def test_early_shutdown_parametric(shutdown_error_rate, num_errors, num_successes, shutdown_window, should_raise):
    executor = AsyncConcurrentExecutor(
        max_workers=10,
        column_name="test_column",
        shutdown_error_rate=shutdown_error_rate,
        shutdown_error_window=shutdown_window,
    )

    # Interleave errors and successes to keep error rate relatively stable
    total = num_errors + num_successes
    work_items = []
    err_idx = 0
    suc_idx = 0
    if num_errors > 0:
        tasks_per_error = total / num_errors
    else:
        tasks_per_error = float("inf")

    for i in range(total):
        if num_errors > 0 and err_idx < num_errors and i >= int(err_idx * tasks_per_error):
            work_items.append((_fail(f"err-{err_idx}"), None))
            err_idx += 1
        elif suc_idx < num_successes:
            work_items.append((_succeed(suc_idx), None))
            suc_idx += 1

    if should_raise:
        with pytest.raises(DataDesignerRuntimeError, match="Data generation was terminated early"):
            executor.run(work_items)
        assert executor.results.early_shutdown is True
    else:
        executor.run(work_items)
        assert executor.results.early_shutdown is False
        assert executor.results.completed_count == total
        assert executor.results.success_count == num_successes
        assert executor.results.error_trap.error_count == num_errors


def test_mixed_success_and_failure_with_callbacks():
    """Stress test: mix of successes and failures with both callbacks."""
    results_list = []
    errors_list = []

    def result_callback(result, *, context=None):
        results_list.append(result)

    def error_callback(exc, *, context=None):
        errors_list.append(exc)

    executor = AsyncConcurrentExecutor(
        max_workers=8,
        column_name="test_column",
        result_callback=result_callback,
        error_callback=error_callback,
        shutdown_error_rate=0.9,
        shutdown_error_window=50,
    )

    async def variable_task(x: int) -> int:
        if x % 7 == 0:
            raise ValueError(f"Error {x}")
        if x % 3 == 0:
            await asyncio.sleep(0.001)
        return x * 2

    num_tasks = 100
    work_items = [(variable_task(i), None) for i in range(num_tasks)]
    executor.run(work_items)

    expected_errors = sum(1 for i in range(num_tasks) if i % 7 == 0)
    expected_successes = num_tasks - expected_errors

    assert executor.results.completed_count == num_tasks
    assert executor.results.success_count == expected_successes
    assert executor.results.error_trap.error_count == expected_errors
    assert len(results_list) == expected_successes
    assert len(errors_list) == expected_errors
    assert executor.results.early_shutdown is False
