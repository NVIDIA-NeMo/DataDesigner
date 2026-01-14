# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading

import pytest

from data_designer.engine.dataset_builders.utils.concurrency import ConcurrentThreadExecutor
from data_designer.engine.errors import DataDesignerRuntimeError


def test_concurrent_thread_executor_wait_for_completion_times_out() -> None:
    gate = threading.Event()

    def blocking_task(ev: threading.Event) -> int:
        ev.wait()
        return 1

    with ConcurrentThreadExecutor(max_workers=1, column_name="test") as executor:
        executor.submit(
            blocking_task,
            gate,
            context={"index": 0},
            acquire_timeout_s=1.0,
            dump_stacks_on_timeout=False,
        )
        with pytest.raises(DataDesignerRuntimeError):
            executor.wait_for_completion(timeout_s=0.01, dump_stacks_on_timeout=False)

        # Ensure the worker can exit so the test suite doesn't retain a stuck thread.
        gate.set()


def test_concurrent_thread_executor_submit_capacity_timeout() -> None:
    gate = threading.Event()

    def blocking_task(ev: threading.Event) -> int:
        ev.wait()
        return 1

    with ConcurrentThreadExecutor(max_workers=1, column_name="test") as executor:
        executor.submit(
            blocking_task,
            gate,
            context={"index": 0},
            acquire_timeout_s=1.0,
            dump_stacks_on_timeout=False,
        )
        with pytest.raises(DataDesignerRuntimeError):
            executor.submit(
                blocking_task,
                gate,
                context={"index": 1},
                acquire_timeout_s=0.01,
                dump_stacks_on_timeout=False,
            )

        gate.set()


