# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from data_designer.slurm.state import SchedulerIdentity
from slurm_test_fakes import FakeCommandResponse, FakeSlurmArray, FakeSlurmRunner, FakeSlurmTask

GOLDEN_DIRECTORY = Path(__file__).parent / "golden" / "slurm"


def _submit(runner: FakeSlurmRunner) -> None:
    completed = runner.run(("sbatch", "--parsable", "run.sbatch"), check=True)
    assert completed.stdout == "4101\n"


def test_fake_slurm_runner_models_array_submission_and_active_states(
    fake_slurm_runner: FakeSlurmRunner,
) -> None:
    _submit(fake_slurm_runner)

    observed = fake_slurm_runner.run(("squeue", "--noheader"), check=True)

    assert observed.stdout == (GOLDEN_DIRECTORY / "squeue_active.txt").read_text()
    assert fake_slurm_runner.calls == [
        ("sbatch", "--parsable", "run.sbatch"),
        ("squeue", "--noheader"),
    ]


def test_fake_slurm_runner_exposes_terminal_accounting_precedence(
    fake_slurm_runner: FakeSlurmRunner,
) -> None:
    _submit(fake_slurm_runner)
    fake_slurm_runner.set_task_state(
        SchedulerIdentity(array_job_id=4101, array_task_id=0),
        queue_state="RUNNING",
        accounting_state="COMPLETED",
    )
    fake_slurm_runner.set_task_state(
        SchedulerIdentity(array_job_id=4101, array_task_id=1),
        queue_state="RUNNING",
        accounting_state="FAILED",
        exit_code="1:0",
    )

    assert "4101_1|RUNNING" in fake_slurm_runner.run(("squeue",)).stdout
    assert fake_slurm_runner.run(("sacct",)).stdout == (GOLDEN_DIRECTORY / "sacct_terminal.txt").read_text()


def test_fake_slurm_runner_models_accounting_lag_and_later_terminal_state(
    fake_slurm_runner: FakeSlurmRunner,
) -> None:
    _submit(fake_slurm_runner)
    scheduler = SchedulerIdentity(array_job_id=4101, array_task_id=0)
    fake_slurm_runner.set_task_state(
        scheduler,
        queue_state=None,
        accounting_state=None,
    )

    assert "4101_0" not in fake_slurm_runner.run(("squeue",)).stdout
    assert "4101_0" not in fake_slurm_runner.run(("sacct",)).stdout

    fake_slurm_runner.set_task_state(
        scheduler,
        queue_state=None,
        accounting_state="COMPLETED",
    )
    assert "4101_0|COMPLETED|0:0" in fake_slurm_runner.run(("sacct",)).stdout


def test_fake_slurm_runner_filters_job_scoped_queries() -> None:
    first = FakeSlurmArray(tasks=(FakeSlurmTask(SchedulerIdentity(array_job_id=4101, array_task_id=0)),))
    second = FakeSlurmArray(
        tasks=(
            FakeSlurmTask(
                SchedulerIdentity(array_job_id=4201, array_task_id=0),
                queue_state="RUNNING",
                accounting_state="FAILED",
                exit_code="1:0",
            ),
        )
    )
    runner = FakeSlurmRunner((first, second))
    runner.run(("sbatch", "first.sbatch"), check=True)
    runner.run(("sbatch", "second.sbatch"), check=True)

    first_queue = runner.run(("squeue", "--jobs", "4101")).stdout
    second_accounting = runner.run(("sacct", "--jobs=4201")).stdout

    assert first_queue == "4101_0|PENDING\n"
    assert second_accounting == "4201_0|FAILED|1:0\n"


@pytest.mark.parametrize(
    "command",
    (
        ("squeue", "--jobs", "--noheader"),
        ("sacct", "--jobs="),
    ),
)
def test_fake_slurm_runner_rejects_malformed_job_selectors(
    fake_slurm_runner: FakeSlurmRunner,
    command: tuple[str, ...],
) -> None:
    _submit(fake_slurm_runner)

    with pytest.raises(AssertionError, match="malformed job selector"):
        fake_slurm_runner.run(command)


@pytest.mark.parametrize("target", ("4101", "4101_1"))
def test_fake_slurm_runner_models_cancellation(
    fake_slurm_runner: FakeSlurmRunner,
    target: str,
) -> None:
    _submit(fake_slurm_runner)

    assert fake_slurm_runner.run(("scancel", target), check=True).returncode == 0
    accounting = fake_slurm_runner.run(("sacct",), check=True).stdout

    expected_tasks = (0, 1) if target == "4101" else (1,)
    for task_id in expected_tasks:
        assert f"4101_{task_id}|CANCELLED|0:15" in accounting


def test_fake_slurm_runner_supports_malformed_output_and_command_failures(
    fake_slurm_runner: FakeSlurmRunner,
) -> None:
    malformed = (GOLDEN_DIRECTORY / "squeue_malformed.txt").read_text()
    fake_slurm_runner.script_next("squeue", FakeCommandResponse(stdout=malformed))
    fake_slurm_runner.script_next("sacct", FakeCommandResponse(stderr="accounting unavailable\n", returncode=2))
    fake_slurm_runner.script_next("sacct", FakeCommandResponse(stderr="accounting unavailable\n", returncode=2))

    assert fake_slurm_runner.run(("squeue",)).stdout == malformed
    assert fake_slurm_runner.run(("sacct",)).returncode == 2
    with pytest.raises(subprocess.CalledProcessError) as error:
        fake_slurm_runner.run(("sacct",), check=True)

    assert error.value.stderr == "accounting unavailable\n"
    fake_slurm_runner.assert_scripts_consumed()


def test_fake_slurm_runner_bounds_sinfo_queries(fake_slurm_runner: FakeSlurmRunner) -> None:
    response = fake_slurm_runner.run(("/usr/bin/sinfo", "--noheader", "--format=%G"), check=True)

    assert response.stdout == (GOLDEN_DIRECTORY / "sinfo_gres.txt").read_text()
    with pytest.raises(AssertionError, match="unexpected sinfo query"):
        fake_slurm_runner.run(("sinfo", "--all"))


def test_fake_slurm_runner_rejects_unscripted_commands_and_submissions(
    fake_slurm_runner: FakeSlurmRunner,
) -> None:
    _submit(fake_slurm_runner)

    assert fake_slurm_runner.run(("sbatch", "second.sbatch")).returncode == 1
    with pytest.raises(AssertionError, match="unexpected command"):
        fake_slurm_runner.run(("srun", "hostname"))
