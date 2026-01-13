# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextvars
import threading
import time
from unittest.mock import Mock

import pytest

from data_designer.engine.dataset_builders.utils.concurrency import (
    ConcurrentThreadExecutor,
    ExecutorResults,
)
from data_designer.engine.errors import DataDesignerRuntimeError, ErrorTrap


@pytest.fixture
def stub_error_trap():
    return ErrorTrap()


@pytest.fixture
def stub_executor_results(stub_error_trap):
    return ExecutorResults(
        failure_threshold=0.1,
        completed_count=10,
        success_count=8,
        early_shutdown=True,
        error_trap=stub_error_trap,
    )


@pytest.fixture
def stub_concurrent_executor():
    return ConcurrentThreadExecutor(max_workers=2, column_name="test_column")


@pytest.mark.parametrize(
    "error_count,completed_count,success_count,window,expected_error_rate",
    [
        (2, 10, 8, 10, 0.2),  # 2 failures out of 10
        (0, 5, 4, 10, 0.0),  # Should return 0 until minimum window is met
        (0, 0, 0, 5, 0.0),  # Zero completed
    ],
)
def test_executor_results_error_rate_calculations(
    error_count, completed_count, success_count, window, expected_error_rate, stub_error_trap
):
    stub_error_trap.error_count = error_count
    results = ExecutorResults(completed_count=completed_count, success_count=success_count, error_trap=stub_error_trap)
    assert results.get_error_rate(window=window) == expected_error_rate


@pytest.mark.parametrize(
    "error_count,failure_threshold,completed_count,window,expected_exceeded",
    [
        (3, 0.2, 10, 10, True),  # 3/10 = 0.3 > 0.2
        (1, 0.2, 10, 10, False),  # 1/10 = 0.1 < 0.2
    ],
)
def test_executor_results_error_rate_exceeded(
    error_count, failure_threshold, completed_count, window, expected_exceeded, stub_error_trap
):
    stub_error_trap.error_count = error_count
    results = ExecutorResults(
        failure_threshold=failure_threshold, completed_count=completed_count, error_trap=stub_error_trap
    )
    assert results.is_error_rate_exceeded(window=window) == expected_exceeded


def test_concurrent_thread_executor_creation():
    executor = ConcurrentThreadExecutor(max_workers=2, column_name="test_column")
    assert executor.max_workers == 2


@pytest.mark.parametrize(
    "task_count,sleep_time,expected_behavior",
    [
        (1, 0, "single_task"),  # Single task
        (5, 0.1, "multiple_tasks"),  # Multiple tasks with sleep
    ],
)
def test_concurrent_thread_executor_submit_tasks(stub_concurrent_executor, task_count, sleep_time, expected_behavior):
    with stub_concurrent_executor as executor:

        def test_func(x):
            if sleep_time > 0:
                time.sleep(sleep_time)  # Simulate some work
            return x * 2

        for i in range(task_count):
            executor.submit(test_func, i)


@pytest.mark.parametrize(
    "shutdown_error_rate,shutdown_error_window,expected_early_shutdown",
    [
        (0.5, 2, True),  # 50% threshold with small window
        (1.0, 10, False),  # 100% threshold with large window
    ],
)
def test_concurrent_thread_executor_early_shutdown_behavior(
    shutdown_error_rate, shutdown_error_window, expected_early_shutdown
):
    executor = ConcurrentThreadExecutor(
        max_workers=2,
        column_name="test_column",
        shutdown_error_rate=shutdown_error_rate,
        shutdown_error_window=shutdown_error_window,
    )

    if expected_early_shutdown:
        with pytest.raises(DataDesignerRuntimeError, match="Data generation was terminated early"):
            with executor:

                def failing_func():
                    raise ValueError("Test error")

                for _ in range(3):
                    executor.submit(failing_func)
                time.sleep(0.1)
    else:
        with executor:

            def failing_func():
                raise ValueError("Test error")

            for _ in range(5):
                executor.submit(failing_func)
            time.sleep(0.1)
            assert executor.results.early_shutdown is False


def test_concurrent_thread_executor_result_callback():
    results = []

    def result_callback(result, *, context=None):
        results.append((result, context))

    with ConcurrentThreadExecutor(
        max_workers=2, column_name="test_column", result_callback=result_callback
    ) as executor:

        def test_func(x):
            return x * 2

        executor.submit(test_func, 5, context={"test": "context"})
        time.sleep(0.1)  # Wait for task to complete

    assert len(results) == 1
    assert results[0][0] == 10
    assert results[0][1] == {"test": "context"}


def test_concurrent_thread_executor_error_callback():
    errors = []

    def error_callback(exc, *, context=None):
        errors.append((exc, context))

    with ConcurrentThreadExecutor(max_workers=2, column_name="test_column", error_callback=error_callback) as executor:

        def failing_func():
            raise ValueError("Test error")

        executor.submit(failing_func, context={"test": "context"})
        time.sleep(0.1)  # Wait for task to complete

    assert len(errors) == 1
    assert isinstance(errors[0][0], ValueError)
    assert errors[0][0].args[0] == "Test error"
    assert errors[0][1] == {"test": "context"}


def test_concurrent_thread_executor_submit_without_context_manager(stub_concurrent_executor):
    with pytest.raises(RuntimeError, match="Executor is not initialized"):
        stub_concurrent_executor.submit(lambda: None)


def test_concurrent_thread_executor_semaphore_behavior(stub_concurrent_executor):
    with stub_concurrent_executor as executor:
        for i in range(2):
            executor.submit(lambda x: time.sleep(0.1), i)
        executor.submit(lambda: None)


@pytest.mark.parametrize(
    "side_effect,expected_exception,expected_message",
    [
        (RuntimeError("Pool shutdown"), DataDesignerRuntimeError, None),
        (RuntimeError("cannot schedule new futures after shutdown"), DataDesignerRuntimeError, None),
        (ValueError("Some error"), ValueError, "Some error"),
    ],
)
def test_concurrent_thread_executor_error_handling_in_submit(
    stub_concurrent_executor, side_effect, expected_exception, expected_message
):
    with stub_concurrent_executor as executor:
        executor._executor.submit = Mock(side_effect=side_effect)
        if expected_message:
            with pytest.raises(expected_exception, match=expected_message):
                executor.submit(lambda: None)
        else:
            with pytest.raises(expected_exception):
                executor.submit(lambda: None)


def test_concurrent_thread_executor_custom_shutdown_parameters():
    executor = ConcurrentThreadExecutor(
        max_workers=2, column_name="test_column", shutdown_error_rate=0.3, shutdown_error_window=5
    )

    assert executor.shutdown_error_rate == 0.3
    assert executor.shutdown_window_size == 5
    assert executor.results.failure_threshold == 0.3
    assert executor.results.early_shutdown is False


def test_disable_early_shutdown_prevents_early_shutdown_raise() -> None:
    executor = ConcurrentThreadExecutor(
        max_workers=2,
        column_name="test_column",
        shutdown_error_rate=0.0,
        shutdown_error_window=0,
        disable_early_shutdown=True,
    )

    with executor:

        def failing_func() -> None:
            raise ValueError("Test error")

        for _ in range(10):
            executor.submit(failing_func)

        deadline = time.time() + 5.0
        while executor.results.completed_count < 10:
            if time.time() > deadline:
                raise AssertionError(
                    f"Timed out waiting for tasks to complete. completed_count={executor.results.completed_count}"
                )
            time.sleep(0.01)

    assert executor.results.error_trap.error_count == 10
    assert executor.results.success_count == 0
    assert executor.results.completed_count == 10
    assert executor.results.early_shutdown is False


def test_disable_early_shutdown_does_not_raise_early_shutdown_error_on_submit_after_shutdown() -> None:
    executor = ConcurrentThreadExecutor(
        max_workers=1,
        column_name="test_column",
        disable_early_shutdown=True,
    )
    with executor:
        executor._executor.submit = Mock(side_effect=RuntimeError("cannot schedule new futures after shutdown"))
        with pytest.raises(RuntimeError, match="cannot schedule new futures after shutdown"):
            executor.submit(lambda: None)


def test_context_variables_context_variable_propagation():
    test_var = contextvars.ContextVar("test_var")
    test_var.set("main_thread_value")

    results = []

    def worker_function():
        results.append(test_var.get())

    with ConcurrentThreadExecutor(max_workers=1, column_name="test_column") as executor:
        executor.submit(worker_function)
        time.sleep(0.1)  # Wait for task to complete

    assert len(results) == 1
    assert results[0] == "main_thread_value"


@pytest.mark.parametrize(
    "max_workers,expected_exception,expected_message",
    [
        (0, ValueError, "max_workers must be greater than 0"),
        (-1, ValueError, "semaphore initial value must be >= 0"),
    ],
)
def test_edge_cases_invalid_max_workers(max_workers, expected_exception, expected_message):
    with pytest.raises(expected_exception, match=expected_message):
        if max_workers == 0:
            with ConcurrentThreadExecutor(max_workers=max_workers, column_name="test_column"):
                pass
        else:
            ConcurrentThreadExecutor(max_workers=max_workers, column_name="test_column")


def test_edge_cases_zero_error_window():
    executor = ConcurrentThreadExecutor(
        max_workers=2, column_name="test_column", shutdown_error_rate=0.5, shutdown_error_window=0
    )

    with pytest.raises(DataDesignerRuntimeError, match="Data generation was terminated early"):
        with executor:

            def failing_func():
                raise ValueError("Test error")

            executor.submit(failing_func)
            time.sleep(0.1)  # Wait for task to complete

            assert executor.results.completed_count == 1


def test_edge_cases_concurrent_submit_calls():
    executor = ConcurrentThreadExecutor(max_workers=4, column_name="test_column")

    def submit_task(task_id):
        def task():
            return f"task_{task_id}"

        executor.submit(task)

    with executor:
        threads = []
        for i in range(10):
            thread = threading.Thread(target=submit_task, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        time.sleep(0.1)  # Wait for tasks to complete


def test_edge_cases_callback_with_none_context():
    results = []
    errors = []

    def result_callback(result, *, context=None):
        results.append((result, context))

    def error_callback(exc, *, context=None):
        errors.append((exc, context))

    with ConcurrentThreadExecutor(
        max_workers=2, column_name="test_column", result_callback=result_callback, error_callback=error_callback
    ) as executor:

        def success_func():
            return "success"

        def failing_func():
            raise ValueError("error")

        executor.submit(success_func, context=None)
        executor.submit(failing_func, context=None)
        time.sleep(0.1)  # Wait for tasks to complete

    assert len(results) == 1
    assert results[0][0] == "success"
    assert results[0][1] is None
    assert len(errors) == 1
    assert isinstance(errors[0][0], ValueError)
    assert errors[0][1] is None


def test_edge_cases_semaphore_release_on_exception():
    executor = ConcurrentThreadExecutor(max_workers=1, column_name="test_column")

    with executor:
        original_release = executor.semaphore.release
        release_count = 0

        def counting_release():
            nonlocal release_count
            release_count += 1
            original_release()

        executor.semaphore.release = counting_release

        def failing_func():
            raise ValueError("Test error")

        executor.submit(failing_func)
        time.sleep(0.1)  # Wait for task to complete

        # Semaphore should have been released
        assert release_count >= 1


def test_edge_cases_multiple_early_shutdown_attempts():
    executor = ConcurrentThreadExecutor(
        max_workers=2,
        column_name="test_column",
        shutdown_error_rate=0.5,  # 50% threshold
        shutdown_error_window=2,  # Small window to trigger quickly
    )

    with pytest.raises(DataDesignerRuntimeError, match="Data generation was terminated early"):
        with executor:

            def failing_func():
                raise ValueError("Test error")

            for _ in range(3):
                executor.submit(failing_func)

            time.sleep(0.1)  # Wait for tasks to complete

            executor.submit(failing_func)


@pytest.mark.parametrize(
    "shutdown_error_rate,num_errors,num_successes,expected_early_shutdown",
    [
        # Threshold 0%: Only succeeds when NO errors
        (0.0, 0, 100, False),  # 0% errors, 0% threshold → No shutdown
        # Threshold below error rate: Should NOT shutdown
        (0.5, 40, 60, False),  # 40% errors, 50% threshold → No shutdown
        (0.3, 20, 80, False),  # 20% errors, 30% threshold → No shutdown
        # Threshold 100%: Only shuts down at exactly 100% errors
        (1.0, 90, 10, False),  # 90% errors, 100% threshold → No shutdown
        (1.0, 50, 50, False),  # 50% errors, 100% threshold → No shutdown
    ],
)
def test_no_early_shutdown_when_below_threshold(
    shutdown_error_rate: float, num_errors: int, num_successes: int, expected_early_shutdown: bool
):
    """Test that early shutdown does NOT occur when error rate is below or at threshold."""
    total_tasks = num_errors + num_successes
    executor = ConcurrentThreadExecutor(
        max_workers=20,
        column_name="test_column",
        shutdown_error_rate=shutdown_error_rate,
        shutdown_error_window=20,
    )

    with executor:

        def success_task(x: int) -> int:
            return x * 2

        def error_task(x: int) -> None:
            raise ValueError(f"Error {x}")

        # Interleave tasks to maintain consistent error rate throughout execution
        # Calculate how many tasks per error to distribute evenly
        if num_errors > 0:
            tasks_per_error = total_tasks / num_errors
        else:
            tasks_per_error = float("inf")

        error_index = 0
        success_index = 0
        for i in range(total_tasks):
            # Determine if this position should be an error based on even distribution
            if num_errors > 0 and error_index < num_errors and i >= int(error_index * tasks_per_error):
                executor.submit(error_task, error_index)
                error_index += 1
            elif success_index < num_successes:
                executor.submit(success_task, success_index)
                success_index += 1

        time.sleep(0.5)  # Wait for all tasks to complete

    # Verify no early shutdown occurred
    assert executor.results.early_shutdown == expected_early_shutdown
    assert executor.results.completed_count == total_tasks
    assert executor.results.success_count == num_successes
    assert executor.results.error_trap.error_count == num_errors


@pytest.mark.parametrize(
    "shutdown_error_rate,num_errors,num_successes,shutdown_window",
    [
        # Threshold 0%: ANY error triggers shutdown
        (0.0, 10, 0, 10),  # 100% errors, 0% threshold → Shutdown
        (0.0, 5, 5, 10),  # 50% errors, 0% threshold → Shutdown
        # Threshold exceeded: Should shutdown
        (0.5, 60, 40, 20),  # 60% errors, 50% threshold → Shutdown
        (0.3, 40, 60, 20),  # 40% errors, 30% threshold → Shutdown
        # Threshold 100%: Shuts down when ALL tasks fail
        (1.0, 20, 0, 10),  # 100% errors, 100% threshold → Shutdown
    ],
)
def test_early_shutdown_when_threshold_exceeded(
    shutdown_error_rate: float, num_errors: int, num_successes: int, shutdown_window: int
):
    """Test that early shutdown DOES occur when error rate exceeds threshold."""
    executor = ConcurrentThreadExecutor(
        max_workers=10,
        column_name="test_column",
        shutdown_error_rate=shutdown_error_rate,
        shutdown_error_window=shutdown_window,
    )

    with pytest.raises(DataDesignerRuntimeError, match="Data generation was terminated early"):
        with executor:

            def success_task(x: int) -> int:
                return x * 2

            def error_task(x: int) -> None:
                raise ValueError(f"Error {x}")

            # Interleave tasks to maintain the specified error rate
            total_tasks = num_errors + num_successes

            # Calculate how many tasks per error to distribute evenly
            if num_errors > 0:
                tasks_per_error = total_tasks / num_errors
            else:
                tasks_per_error = float("inf")

            error_index = 0
            success_index = 0
            for i in range(total_tasks):
                # Determine if this position should be an error based on even distribution
                if num_errors > 0 and error_index < num_errors and i >= int(error_index * tasks_per_error):
                    executor.submit(error_task, error_index)
                    error_index += 1
                elif success_index < num_successes:
                    executor.submit(success_task, success_index)
                    success_index += 1

            time.sleep(0.3)

            # Try to submit more tasks - should raise because early_shutdown is True
            executor.submit(success_task, 999)


def test_thread_safety_with_high_concurrency():
    """Stress test to verify thread-safe counter updates under high concurrency."""
    num_tasks = 500
    max_workers = 20
    results_list = []
    errors_list = []

    def result_callback(result: int, *, context: dict | None = None) -> None:
        results_list.append(result)

    def error_callback(exc: Exception, *, context: dict | None = None) -> None:
        errors_list.append(exc)

    with ConcurrentThreadExecutor(
        max_workers=max_workers,
        column_name="test_column",
        result_callback=result_callback,
        error_callback=error_callback,
        shutdown_error_rate=0.9,  # High threshold to avoid shutdown
        shutdown_error_window=50,
    ) as executor:

        def variable_task(x: int) -> int:
            # Some tasks succeed, some fail, some take longer
            if x % 7 == 0:
                raise ValueError(f"Error {x}")
            if x % 3 == 0:
                time.sleep(0.001)
            return x * 2

        for i in range(num_tasks):
            executor.submit(variable_task, i)

        time.sleep(1.0)  # Wait for all tasks to complete

    # Calculate expected counts
    expected_errors = sum(1 for i in range(num_tasks) if i % 7 == 0)
    expected_successes = num_tasks - expected_errors

    # Verify all counts are accurate (no race conditions)
    assert executor.results.completed_count == num_tasks
    assert executor.results.success_count == expected_successes
    assert executor.results.error_trap.error_count == expected_errors
    assert len(results_list) == expected_successes
    assert len(errors_list) == expected_errors
    assert executor.results.early_shutdown is False
