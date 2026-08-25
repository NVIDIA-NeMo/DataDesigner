# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from collections.abc import Sequence

import pytest
from slurm_test_fakes import FakeCommandResponse, FakeSlurmRunner

from data_designer.slurm.launcher import SlurmCommandClient, SlurmCommandError, SlurmExecutables, SlurmParseError
from data_designer.slurm.state import SchedulerIdentity, SchedulerState


def test_client_submits_and_observes_one_managed_array(fake_slurm_runner: FakeSlurmRunner) -> None:
    client = SlurmCommandClient(fake_slurm_runner)

    submission = client.submit("/workspace/run.sbatch")
    queue = client.query_queue((submission.array_job_id,))

    assert submission.array_job_id == 4101
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


def test_client_rejects_unrequested_scheduler_records(fake_slurm_runner: FakeSlurmRunner) -> None:
    client = SlurmCommandClient(fake_slurm_runner)
    fake_slurm_runner.script_next("squeue", FakeCommandResponse(stdout="9999_0|RUNNING\n"))

    with pytest.raises(SlurmParseError, match="unrequested"):
        client.query_queue((4101,))

    fake_slurm_runner.script_next("sacct", FakeCommandResponse(stdout="9999_0|FAILED|1:0\n"))
    with pytest.raises(SlurmParseError, match="unrequested"):
        client.query_accounting((SchedulerIdentity(array_job_id=4101, array_task_id=0),))


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


def test_client_rejects_non_text_runner_output() -> None:
    client = SlurmCommandClient(_NonTextRunner())

    with pytest.raises(SlurmCommandError, match="did not return text output"):
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
    def run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        completed = subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")
        completed.stdout = b"not text"  # type: ignore[assignment]
        return completed
