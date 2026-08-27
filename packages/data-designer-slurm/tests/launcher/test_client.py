# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from collections.abc import Sequence

import pytest
from slurm_test_fakes import FakeCommandResponse, FakeSlurmJob, FakeSlurmRunner

from data_designer.slurm.launcher.client import SlurmCommandClient, SlurmExecutables
from data_designer.slurm.launcher.errors import SlurmCommandError, SlurmCommandOutputError
from data_designer.slurm.state import SchedulerIdentity, SchedulerState


def test_client_submits_and_observes_one_managed_array(fake_slurm_runner: FakeSlurmRunner) -> None:
    client = SlurmCommandClient(fake_slurm_runner)

    submission = client.submit("/workspace/run.sbatch")
    queue = client.query_queue((submission.job_id,))

    assert submission.job_id == 4101
    assert tuple(record.state for record in queue) == (SchedulerState.PENDING, SchedulerState.RUNNING)
    assert fake_slurm_runner.calls == [
        ("sbatch", "--parsable", "--export=NIL", "/workspace/run.sbatch"),
        ("squeue", "--noheader", "--array", "--format=%i|%T", "--jobs=4101"),
    ]


def test_client_queries_accounting_and_cancels_one_array_task(fake_slurm_runner: FakeSlurmRunner) -> None:
    client = SlurmCommandClient(fake_slurm_runner)
    client.submit("run.sbatch")
    scheduler = SchedulerIdentity(array_job_id=4101, array_task_id=1)

    client.cancel(scheduler)
    accounting = client.query_accounting((scheduler,))

    assert len(accounting) == 1
    assert accounting[0].job_identity == scheduler
    assert accounting[0].state is SchedulerState.CANCELLED
    assert fake_slurm_runner.calls[-2:] == [
        ("scancel", "4101_1"),
        (
            "sacct",
            "--noheader",
            "--array",
            "--allocations",
            "--parsable2",
            "--format=JobID,State,ExitCode",
            "--jobs=4101_1",
        ),
    ]


def test_client_deduplicates_explicit_job_selectors(fake_slurm_runner: FakeSlurmRunner) -> None:
    client = SlurmCommandClient(fake_slurm_runner)
    client.submit("run.sbatch")

    client.query_queue((4101, 4101, SchedulerIdentity(array_job_id=4101, array_task_id=0)))

    assert fake_slurm_runner.calls[-1][-1] == "--jobs=4101,4101_0"


def test_client_rejects_unrequested_scheduler_records(fake_slurm_runner: FakeSlurmRunner) -> None:
    client = SlurmCommandClient(fake_slurm_runner)
    fake_slurm_runner.script_next("squeue", FakeCommandResponse(stdout="9999_0|RUNNING\n"))

    with pytest.raises(SlurmCommandOutputError, match="unrequested"):
        client.query_queue((4101,))

    fake_slurm_runner.script_next("sacct", FakeCommandResponse(stdout="9999_0|FAILED|1:0\n"))
    with pytest.raises(SlurmCommandOutputError, match="unrequested"):
        client.query_accounting((SchedulerIdentity(array_job_id=4101, array_task_id=0),))


def test_client_observes_regular_cpu_job() -> None:
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(job_id=5101),))
    client = SlurmCommandClient(runner)

    submission = client.submit("image-build.sbatch")
    queue = client.query_queue((5101,))
    runner.set_job_state(5101, queue_state=None, accounting_state="COMPLETED")
    accounting = client.query_accounting((5101,))

    assert submission.job_id == 5101
    assert queue[0].job_identity == 5101
    assert queue[0].state is SchedulerState.PENDING
    assert accounting[0].job_identity == 5101
    assert accounting[0].state is SchedulerState.COMPLETED


def test_client_ignores_array_parent_observation_for_exact_task() -> None:
    runner = FakeSlurmRunner()
    runner.script_next("sacct", FakeCommandResponse(stdout="4101|RUNNING|0:0\n"))
    client = SlurmCommandClient(runner)

    records = client.query_accounting((SchedulerIdentity(array_job_id=4101, array_task_id=0),))

    assert records == ()


def test_client_keeps_explicitly_selected_parent_observation() -> None:
    runner = FakeSlurmRunner()
    runner.script_next("sacct", FakeCommandResponse(stdout="4101|RUNNING|0:0\n"))
    client = SlurmCommandClient(runner)
    task = SchedulerIdentity(array_job_id=4101, array_task_id=0)

    records = client.query_accounting((4101, task))

    assert len(records) == 1
    assert records[0].job_identity == 4101


def test_client_rejects_unbounded_or_invalid_job_selectors(fake_slurm_runner: FakeSlurmRunner) -> None:
    client = SlurmCommandClient(fake_slurm_runner)

    with pytest.raises(ValueError, match="at least one"):
        client.query_queue(())
    with pytest.raises(ValueError, match="positive 32-bit integers"):
        client.query_accounting((0,))
    with pytest.raises(ValueError, match="positive 32-bit integers"):
        client.cancel(True)
    with pytest.raises(ValueError, match="32-bit"):
        client.cancel(1 << 32)
    with pytest.raises(ValueError, match="array-task IDs"):
        client.cancel(SchedulerIdentity(array_job_id=4101, array_task_id=1 << 32))

    assert fake_slurm_runner.calls == []


def test_client_queries_bounded_gpu_inventory(fake_slurm_runner: FakeSlurmRunner) -> None:
    client = SlurmCommandClient(fake_slurm_runner)

    assert client.query_gpu_counts() == (2,)
    assert fake_slurm_runner.calls == [("sinfo", "--noheader", "--format=%G")]


def test_client_queries_partition_scoped_gpu_inventory() -> None:
    command = ("sinfo", "--noheader", "--format=%G", "--partition=batch")
    runner = FakeSlurmRunner(sinfo_responses={command: FakeCommandResponse(stdout="gpu:a100:8\n")})

    assert SlurmCommandClient(runner).query_gpu_counts(partition="batch") == (8,)
    assert runner.calls == [command]


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


def test_client_bounds_command_failure_detail(fake_slurm_runner: FakeSlurmRunner) -> None:
    fake_slurm_runner.script_next("squeue", FakeCommandResponse(stderr="x" * 600, returncode=2))
    client = SlurmCommandClient(fake_slurm_runner)

    with pytest.raises(SlurmCommandError) as error:
        client.query_queue((4101,))

    detail = str(error.value).partition(": ")[2]
    assert len(detail) == 512
    assert detail.endswith("...")


def test_client_normalizes_execution_errors() -> None:
    client = SlurmCommandClient(_FailingRunner())

    with pytest.raises(SlurmCommandError, match="squeue could not be executed") as error:
        client.query_queue((4101,))

    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_client_normalizes_command_timeouts() -> None:
    client = SlurmCommandClient(_TimeoutRunner())

    with pytest.raises(SlurmCommandError, match="command timed out") as error:
        client.query_queue((4101,))

    assert isinstance(error.value.__cause__, subprocess.TimeoutExpired)


@pytest.mark.parametrize("returncode", (0, 2))
def test_client_rejects_non_text_runner_output(returncode: int) -> None:
    client = SlurmCommandClient(_NonTextRunner(returncode))

    with pytest.raises(SlurmCommandError, match="malformed process result"):
        client.query_queue((4101,))


def test_script_path_is_one_argument_vector_token(fake_slurm_runner: FakeSlurmRunner) -> None:
    client = SlurmCommandClient(fake_slurm_runner)

    client.submit("/workspace/run; touch injected.sbatch")

    assert fake_slurm_runner.calls[0] == (
        "sbatch",
        "--parsable",
        "--export=NIL",
        "/workspace/run; touch injected.sbatch",
    )


def test_client_rejects_option_like_script_path(fake_slurm_runner: FakeSlurmRunner) -> None:
    client = SlurmCommandClient(fake_slurm_runner)

    with pytest.raises(ValueError, match="must not begin"):
        client.submit("--wrap=unexpected")

    assert fake_slurm_runner.calls == []


@pytest.mark.parametrize("script_path", ("", "bad\npath"))
def test_client_rejects_invalid_script_path(fake_slurm_runner: FakeSlurmRunner, script_path: str) -> None:
    client = SlurmCommandClient(fake_slurm_runner)

    with pytest.raises(ValueError, match="batch script path"):
        client.submit(script_path)

    assert fake_slurm_runner.calls == []


@pytest.mark.parametrize("executable", ("", "sbatch --wait", "sbatch\n"))
def test_executables_reject_invalid_tokens(executable: str) -> None:
    with pytest.raises(ValueError, match="Slurm executable"):
        SlurmExecutables(sbatch=executable)


class _FailingRunner:
    def run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        del command
        raise FileNotFoundError("missing executable")


class _TimeoutRunner:
    def run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, 30.0)


class _NonTextRunner:
    def __init__(self, returncode: int) -> None:
        self._returncode = returncode

    def run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        completed = subprocess.CompletedProcess(command, self._returncode, stdout="ok", stderr="")
        if self._returncode:
            completed.stderr = b"not text"  # type: ignore[assignment]
        else:
            completed.stdout = b"not text"  # type: ignore[assignment]
        return completed
