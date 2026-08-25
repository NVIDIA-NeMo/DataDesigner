# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from collections.abc import Sequence

import pytest
from slurm_test_fakes import FakeCommandResponse, FakeSlurmRunner

from data_designer.slurm.launcher import SlurmCommandClient, SlurmCommandError
from data_designer.slurm.state import SchedulerIdentity, SchedulerState


def test_client_submits_and_observes_one_managed_array(fake_slurm_runner: FakeSlurmRunner) -> None:
    client = SlurmCommandClient(fake_slurm_runner)

    submission = client.submit("/workspace/run.sbatch")
    queue = client.query_queue((submission.array_job_id,))

    assert submission.array_job_id == 4101
    assert tuple(record.state for record in queue) == (SchedulerState.PENDING, SchedulerState.RUNNING)
    assert fake_slurm_runner.calls == [
        ("sbatch", "--parsable", "/workspace/run.sbatch"),
        ("squeue", "--noheader", "--array", "--format=%i|%T", "--jobs=4101"),
    ]


def test_client_queries_accounting_and_cancels_one_array_task(fake_slurm_runner: FakeSlurmRunner) -> None:
    client = SlurmCommandClient(fake_slurm_runner)
    client.submit("run.sbatch")
    scheduler = SchedulerIdentity(array_job_id=4101, array_task_id=1)

    client.cancel(scheduler)
    accounting = client.query_accounting((scheduler,))

    assert len(accounting) == 1
    assert accounting[0].scheduler == scheduler
    assert accounting[0].state is SchedulerState.CANCELLED
    assert fake_slurm_runner.calls[-2:] == [
        ("scancel", "4101_1"),
        (
            "sacct",
            "--noheader",
            "--array",
            "--allocations",
            "--parsable2",
            "--format=JobIDRaw,State,ExitCode",
            "--jobs=4101_1",
        ),
    ]


def test_client_deduplicates_explicit_job_selectors(fake_slurm_runner: FakeSlurmRunner) -> None:
    client = SlurmCommandClient(fake_slurm_runner)
    client.submit("run.sbatch")

    client.query_queue((4101, 4101, SchedulerIdentity(array_job_id=4101, array_task_id=0)))

    assert fake_slurm_runner.calls[-1][-1] == "--jobs=4101,4101_0"


def test_client_rejects_unbounded_or_invalid_job_selectors(fake_slurm_runner: FakeSlurmRunner) -> None:
    client = SlurmCommandClient(fake_slurm_runner)

    with pytest.raises(ValueError, match="at least one"):
        client.query_queue(())
    with pytest.raises(ValueError, match="positive integers"):
        client.query_accounting((0,))
    with pytest.raises(ValueError, match="positive integers"):
        client.cancel(True)

    assert fake_slurm_runner.calls == []


def test_client_queries_bounded_gpu_inventory(fake_slurm_runner: FakeSlurmRunner) -> None:
    client = SlurmCommandClient(fake_slurm_runner)

    assert client.query_gpu_counts() == (2,)
    assert fake_slurm_runner.calls == [("sinfo", "--noheader", "--format=%G")]


def test_client_rejects_invalid_gpu_partition_without_running_command(fake_slurm_runner: FakeSlurmRunner) -> None:
    client = SlurmCommandClient(fake_slurm_runner)

    with pytest.raises(ValueError, match="valid identifier"):
        client.query_gpu_counts(partition="batch,other")

    assert fake_slurm_runner.calls == []


def test_client_normalizes_command_failures(fake_slurm_runner: FakeSlurmRunner) -> None:
    fake_slurm_runner.script_next(
        "sacct",
        FakeCommandResponse(stderr="accounting\nservice unavailable\n", returncode=2),
    )
    client = SlurmCommandClient(fake_slurm_runner)

    with pytest.raises(SlurmCommandError, match="sacct failed with exit code 2: accounting service unavailable"):
        client.query_accounting((4101,))


def test_client_removes_terminal_controls_from_command_failures(fake_slurm_runner: FakeSlurmRunner) -> None:
    fake_slurm_runner.script_next(
        "squeue",
        FakeCommandResponse(stderr="queue unavailable\x1b[31m\n", returncode=2),
    )
    client = SlurmCommandClient(fake_slurm_runner)

    with pytest.raises(SlurmCommandError) as error:
        client.query_queue((4101,))

    assert "\x1b" not in str(error.value)


def test_client_normalizes_execution_errors() -> None:
    client = SlurmCommandClient(_FailingRunner())

    with pytest.raises(SlurmCommandError, match="squeue could not be executed") as error:
        client.query_queue((4101,))

    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_script_path_is_one_argument_vector_token(fake_slurm_runner: FakeSlurmRunner) -> None:
    client = SlurmCommandClient(fake_slurm_runner)

    client.submit("/workspace/run; touch injected.sbatch")

    assert fake_slurm_runner.calls[0] == (
        "sbatch",
        "--parsable",
        "/workspace/run; touch injected.sbatch",
    )


def test_client_rejects_option_like_script_path(fake_slurm_runner: FakeSlurmRunner) -> None:
    client = SlurmCommandClient(fake_slurm_runner)

    with pytest.raises(ValueError, match="must not begin"):
        client.submit("--wrap=unexpected")

    assert fake_slurm_runner.calls == []


class _FailingRunner:
    def run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        del command
        raise FileNotFoundError("missing executable")
