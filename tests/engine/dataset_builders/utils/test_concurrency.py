# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    assert executor._max_workers == 2


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
            assert executor._results.early_shutdown is False


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

    assert executor._shutdown_error_rate == 0.3
    assert executor._shutdown_window_size == 5
    assert executor._results.failure_threshold == 0.3


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

            assert executor._results.completed_count == 1


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
        original_release = executor._semaphore.release
        release_count = 0

        def counting_release():
            nonlocal release_count
            release_count += 1
            original_release()

        executor._semaphore.release = counting_release

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
