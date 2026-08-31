# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import re
import subprocess
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from data_designer.slurm.state import SchedulerIdentity

_JOB_SELECTOR_PATTERN = re.compile(r"^[0-9]+(?:_[0-9]+)?$")
_SQUEUE_REQUIRED_ARGUMENTS = ("--noheader", "--array", "--format=%i|%T")
_SACCT_REQUIRED_ARGUMENTS = (
    "--noheader",
    "--array",
    "--allocations",
    "--parsable2",
    "--format=JobID,State,ExitCode",
)


@dataclass(frozen=True)
class FakeCommandResponse:
    """One deterministic subprocess response."""

    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


@dataclass
class FakeSlurmJob:
    """Mutable scheduler views for one ordinary batch-job identity."""

    job_id: int
    queue_state: str | None = "PENDING"
    accounting_state: str | None = None
    exit_code: str = "0:0"

    def __post_init__(self) -> None:
        if type(self.job_id) is not int or self.job_id <= 0:
            raise ValueError("fake Slurm job IDs must be positive integers")


@dataclass
class FakeSlurmTask:
    """Mutable scheduler views for one canonical array-task identity."""

    scheduler: SchedulerIdentity
    queue_state: str | None = "PENDING"
    accounting_state: str | None = None
    exit_code: str = "0:0"


@dataclass
class FakeSlurmArray:
    """A deterministic array submission exposed through Slurm command output."""

    tasks: tuple[FakeSlurmTask, ...]

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("fake Slurm arrays require at least one task")
        job_ids = {task.scheduler.array_job_id for task in self.tasks}
        task_ids = [task.scheduler.array_task_id for task in self.tasks]
        if len(job_ids) != 1:
            raise ValueError("fake Slurm array tasks must share one array job ID")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("fake Slurm array task IDs must be unique")

    @property
    def array_job_id(self) -> int:
        """Return the canonical job ID shared by the array tasks."""
        return self.tasks[0].scheduler.array_job_id


class FakeSlurmRunner:
    """Stateful fake that copies configured jobs before exposing Slurm commands."""

    def __init__(
        self,
        arrays: Iterable[FakeSlurmArray] = (),
        *,
        jobs: Iterable[FakeSlurmJob] = (),
        sinfo_responses: Mapping[tuple[str, ...], FakeCommandResponse] | None = None,
    ) -> None:
        submissions = (*arrays, *jobs)
        job_ids = tuple(
            submission.array_job_id if isinstance(submission, FakeSlurmArray) else submission.job_id
            for submission in submissions
        )
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("fake Slurm submissions must have unique job IDs")
        self._pending_submissions = deque(copy.deepcopy(submissions))
        self._submitted_arrays: dict[int, FakeSlurmArray] = {}
        self._submitted_jobs: dict[int, FakeSlurmJob] = {}
        self._scripted_responses: dict[str, deque[FakeCommandResponse]] = {}
        self._sinfo_responses = dict(sinfo_responses or {})
        self.calls: list[tuple[str, ...]] = []
        self.inputs: list[str | None] = []

    def run(
        self,
        command: Sequence[str],
        *,
        check: bool = False,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run one fake Slurm command and optionally raise on failure."""
        if not command:
            raise ValueError("command must not be empty")
        argv = tuple(command)
        self.calls.append(argv)
        self.inputs.append(input_text)
        command_name = Path(argv[0]).name
        scripted = self._scripted_responses.get(command_name)
        if scripted:
            response = scripted.popleft()
        else:
            response = self._dispatch(command_name, argv)

        completed = subprocess.CompletedProcess(
            args=list(argv),
            returncode=response.returncode,
            stdout=response.stdout,
            stderr=response.stderr,
        )
        if check and response.returncode:
            raise subprocess.CalledProcessError(
                response.returncode,
                list(argv),
                output=response.stdout,
                stderr=response.stderr,
            )
        return completed

    def script_next(self, command_name: str, response: FakeCommandResponse) -> None:
        """Inject one explicit response before the command's stateful behavior."""
        self._scripted_responses.setdefault(command_name, deque()).append(response)

    def set_task_state(
        self,
        scheduler: SchedulerIdentity,
        *,
        queue_state: str | None,
        accounting_state: str | None,
        exit_code: str = "0:0",
    ) -> None:
        """Set the independently observable queue and accounting states."""
        task = self._find_task(scheduler)
        task.queue_state = queue_state
        task.accounting_state = accounting_state
        task.exit_code = exit_code

    def set_job_state(
        self,
        job_id: int,
        *,
        queue_state: str | None,
        accounting_state: str | None,
        exit_code: str = "0:0",
    ) -> None:
        """Set the independently observable states for one ordinary job."""
        job = self._submitted_jobs[job_id]
        job.queue_state = queue_state
        job.accounting_state = accounting_state
        job.exit_code = exit_code

    def assert_scripts_consumed(self) -> None:
        """Assert that no scripted command response remains."""
        remaining = sum(len(responses) for responses in self._scripted_responses.values())
        if remaining:
            raise AssertionError(f"{remaining} scripted Slurm responses remain")

    def _dispatch(self, command_name: str, argv: tuple[str, ...]) -> FakeCommandResponse:
        handlers = {
            "sacct": self._run_sacct,
            "sbatch": self._run_sbatch,
            "scancel": self._run_scancel,
            "sinfo": self._run_sinfo,
            "squeue": self._run_squeue,
        }
        try:
            handler = handlers[command_name]
        except KeyError:
            raise AssertionError(f"unexpected command {command_name!r}") from None
        return handler(argv)

    def _run_sbatch(self, argv: tuple[str, ...]) -> FakeCommandResponse:
        if not self._pending_submissions:
            return FakeCommandResponse(stderr="no scripted submission\n", returncode=1)
        submission = self._pending_submissions.popleft()
        if isinstance(submission, FakeSlurmArray):
            job_id = submission.array_job_id
            self._submitted_arrays[job_id] = submission
        else:
            job_id = submission.job_id
            self._submitted_jobs[job_id] = submission
        if "--parsable" in argv[1:]:
            return FakeCommandResponse(stdout=f"{job_id}\n")
        return FakeCommandResponse(stdout=f"Submitted batch job {job_id}\n")

    def _run_squeue(self, argv: tuple[str, ...]) -> FakeCommandResponse:
        self._require_arguments(argv, _SQUEUE_REQUIRED_ARGUMENTS)
        rows = [
            f"{job.job_id}|{job.queue_state}"
            for job in self._selected_submitted_jobs(argv)
            if job.queue_state is not None
        ] + [
            f"{task.scheduler.array_job_id}_{task.scheduler.array_task_id}|{task.queue_state}"
            for task in self._selected_submitted_tasks(argv)
            if task.queue_state is not None
        ]
        return FakeCommandResponse(stdout="".join(f"{row}\n" for row in sorted(rows)))

    def _run_sacct(self, argv: tuple[str, ...]) -> FakeCommandResponse:
        self._require_arguments(argv, _SACCT_REQUIRED_ARGUMENTS)
        rows = [
            f"{job.job_id}|{job.accounting_state}|{job.exit_code}"
            for job in self._selected_submitted_jobs(argv)
            if job.accounting_state is not None
        ] + [
            (f"{task.scheduler.array_job_id}_{task.scheduler.array_task_id}|{task.accounting_state}|{task.exit_code}")
            for task in self._selected_submitted_tasks(argv)
            if task.accounting_state is not None
        ]
        return FakeCommandResponse(stdout="".join(f"{row}\n" for row in sorted(rows)))

    def _run_scancel(self, argv: tuple[str, ...]) -> FakeCommandResponse:
        targets = tuple(argument for argument in argv[1:] if not argument.startswith("-"))
        if len(targets) != 1:
            return FakeCommandResponse(stderr="expected one cancellation target\n", returncode=1)
        target = targets[0]
        if "_" not in target and target.isdecimal() and int(target) in self._submitted_jobs:
            job = self._submitted_jobs[int(target)]
            job.queue_state = None
            job.accounting_state = "CANCELLED"
            job.exit_code = "0:15"
            return FakeCommandResponse()
        try:
            tasks = self._tasks_for_target(target)
        except (KeyError, ValueError):
            return FakeCommandResponse(stderr="unknown cancellation target\n", returncode=1)
        for task in tasks:
            task.queue_state = None
            task.accounting_state = "CANCELLED"
            task.exit_code = "0:15"
        return FakeCommandResponse()

    def _run_sinfo(self, argv: tuple[str, ...]) -> FakeCommandResponse:
        key = ("sinfo", *argv[1:])
        try:
            return self._sinfo_responses[key]
        except KeyError:
            raise AssertionError(f"unexpected sinfo query {key!r}") from None

    @staticmethod
    def _require_arguments(argv: tuple[str, ...], required: tuple[str, ...]) -> None:
        missing = tuple(argument for argument in required if argument not in argv[1:])
        if missing:
            raise AssertionError(f"missing required arguments {missing!r} in {argv!r}")

    def _tasks_for_target(self, target: str) -> tuple[FakeSlurmTask, ...]:
        if "_" not in target:
            return self._submitted_arrays[int(target)].tasks
        job_id, task_id = target.split("_", 1)
        scheduler = SchedulerIdentity(array_job_id=int(job_id), array_task_id=int(task_id))
        return (self._find_task(scheduler),)

    def _find_task(self, scheduler: SchedulerIdentity) -> FakeSlurmTask:
        array = self._submitted_arrays[scheduler.array_job_id]
        for task in array.tasks:
            if task.scheduler == scheduler:
                return task
        raise KeyError(scheduler)

    def _sorted_submitted_tasks(self) -> list[FakeSlurmTask]:
        return sorted(
            (task for array in self._submitted_arrays.values() for task in array.tasks),
            key=lambda task: (task.scheduler.array_job_id, task.scheduler.array_task_id),
        )

    def _selected_submitted_tasks(self, argv: tuple[str, ...]) -> list[FakeSlurmTask]:
        selectors = self._job_selectors(argv)
        if not selectors:
            return self._sorted_submitted_tasks()

        selected: dict[SchedulerIdentity, FakeSlurmTask] = {}
        for selector in selectors:
            try:
                tasks = self._tasks_for_target(selector)
            except KeyError:
                continue
            selected.update((task.scheduler, task) for task in tasks)
        return sorted(
            selected.values(),
            key=lambda task: (task.scheduler.array_job_id, task.scheduler.array_task_id),
        )

    def _selected_submitted_jobs(self, argv: tuple[str, ...]) -> list[FakeSlurmJob]:
        selectors = self._job_selectors(argv)
        if not selectors:
            return sorted(self._submitted_jobs.values(), key=lambda job: job.job_id)
        return sorted(
            (
                self._submitted_jobs[int(selector)]
                for selector in selectors
                if "_" not in selector and int(selector) in self._submitted_jobs
            ),
            key=lambda job: job.job_id,
        )

    @staticmethod
    def _job_selectors(argv: tuple[str, ...]) -> tuple[str, ...]:
        selectors: list[str] = []
        for index, argument in enumerate(argv[1:]):
            if argument in {"-j", "--jobs"}:
                try:
                    selectors.extend(argv[index + 2].split(","))
                except IndexError:
                    raise AssertionError(f"missing job selector in {argv!r}") from None
            elif argument.startswith("--jobs="):
                selectors.extend(argument.partition("=")[2].split(","))
        if not selectors:
            return ()
        for selector in selectors:
            if _JOB_SELECTOR_PATTERN.fullmatch(selector) is None:
                raise AssertionError(f"malformed job selector {selector!r} in {argv!r}")
        return tuple(selectors)
