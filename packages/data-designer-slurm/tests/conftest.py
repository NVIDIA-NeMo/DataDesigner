# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from slurm_test_fakes import (
    FakeClock,
    FakeCommandResponse,
    FakeLogicalEndpoint,
    FakeSlurmArray,
    FakeSlurmRunner,
    FakeSlurmTask,
    FakeVllmBackend,
)

from data_designer.slurm.state import SchedulerIdentity

TEST_DIRECTORY = Path(__file__).parent
SLURM_GOLDEN_DIRECTORY = TEST_DIRECTORY / "slurm_test_fakes" / "golden" / "slurm"


@pytest.fixture
def fake_clock() -> FakeClock:
    """Return an isolated explicitly controlled clock."""
    return FakeClock(datetime(2026, 8, 18, 12, tzinfo=timezone.utc), monotonic_time=100.0)


@pytest.fixture
def fake_slurm_array() -> FakeSlurmArray:
    """Return a two-task array copied when the fake runner is constructed."""
    return FakeSlurmArray(
        tasks=(
            FakeSlurmTask(SchedulerIdentity(array_job_id=4101, array_task_id=0)),
            FakeSlurmTask(
                SchedulerIdentity(array_job_id=4101, array_task_id=1),
                queue_state="RUNNING",
            ),
        )
    )


@pytest.fixture
def fake_slurm_runner(fake_slurm_array: FakeSlurmArray) -> FakeSlurmRunner:
    """Return an isolated Slurm runner with one array and one bounded sinfo query."""
    return FakeSlurmRunner(
        arrays=(fake_slurm_array,),
        sinfo_responses={
            ("sinfo", "--noheader", "--format=%G"): FakeCommandResponse(
                stdout=(SLURM_GOLDEN_DIRECTORY / "sinfo_gres.txt").read_text()
            )
        },
    )


@pytest.fixture
def fake_plugin_overlay() -> Path:
    """Return the installed-layout fake Data Designer plugin overlay."""
    return TEST_DIRECTORY / "fixtures" / "fake_plugin_overlay"


@pytest.fixture
def fake_logical_endpoint() -> FakeLogicalEndpoint:
    """Return an isolated two-backend logical endpoint."""
    return FakeLogicalEndpoint(
        "http://127.0.0.1:31000",
        (
            FakeVllmBackend("http://127.0.0.1:31001", rank=0),
            FakeVllmBackend("http://127.0.0.1:31002", rank=1),
        ),
    )
