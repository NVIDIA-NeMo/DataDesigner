# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.engine.dataset_builders.scheduling.resources import (
    SchedulableTask,
    SchedulerResourceRequest,
    TaskGroupKey,
    TaskGroupSpec,
    stable_task_id,
)
from data_designer.engine.dataset_builders.scheduling.task_model import Task


def test_scheduler_resource_request_defaults_to_submission() -> None:
    request = SchedulerResourceRequest()

    assert request.amounts == {"submission": 1}


def test_scheduler_resource_request_accepts_dynamic_resource_keys() -> None:
    request = SchedulerResourceRequest({"request:nvidia/nemotron": 1})

    assert request.amounts == {"request:nvidia/nemotron": 1}


def test_scheduler_resource_request_rejects_empty_resource_key() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        SchedulerResourceRequest({"": 1})


def test_scheduler_resource_request_rejects_non_positive_amounts() -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        SchedulerResourceRequest({"submission": 0})


def test_stable_task_id_is_stable_for_task_identity() -> None:
    task = Task(column="answer", row_group=3, row_index=8, task_type="cell")

    assert stable_task_id(task) == stable_task_id(task)
    assert stable_task_id(task).startswith("task-")


def test_stable_task_id_distinguishes_task_identity_fields() -> None:
    first = Task(column="answer", row_group=3, row_index=8, task_type="cell")
    second = Task(column="answer", row_group=3, row_index=9, task_type="cell")

    assert stable_task_id(first) != stable_task_id(second)


def test_schedulable_task_binds_payload_group_and_resource_request() -> None:
    task = Task(column="answer", row_group=0, row_index=1, task_type="cell")
    group = TaskGroupSpec(TaskGroupKey(kind="model", identity=("nvidia", "nemotron")), admitted_limit=2)
    request = SchedulerResourceRequest({"submission": 1, "llm_wait": 1})

    item = SchedulableTask(
        task_id=stable_task_id(task),
        payload=task,
        group=group,
        resource_request=request,
    )

    assert item.payload == task
    assert item.group == group
    assert item.resource_request.amounts["llm_wait"] == 1
