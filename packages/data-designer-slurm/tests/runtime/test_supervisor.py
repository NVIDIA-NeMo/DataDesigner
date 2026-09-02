# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from slurm_test_fakes import FakeClock

from data_designer.slurm.runtime.errors import SlurmRuntimeError
from data_designer.slurm.runtime.models import RuntimeStep, RuntimeStepRole
from data_designer.slurm.runtime.supervisor import StepSupervisor, SubprocessStepRunner


@dataclass(slots=True)
class _Process:
    pid: int
    returncode: int | None = None
    terminated: int = 0
    killed: int = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated += 1
        self.returncode = -15

    def kill(self) -> None:
        self.killed += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        assert self.returncode is not None
        return self.returncode


class _Runner:
    def __init__(self, processes: list[_Process]) -> None:
        self.processes = processes
        self.started = 0

    def start(self, step: RuntimeStep) -> _Process:
        del step
        process = self.processes[self.started]
        self.started += 1
        return process


def test_cleanup_is_reverse_order_and_idempotent(tmp_path: Path, fake_clock: FakeClock) -> None:
    first = _Process(1)
    second = _Process(2)
    runner = _Runner([first, second])
    supervisor = StepSupervisor(runner, clock=fake_clock)
    supervisor.start(_step(tmp_path, "first"))
    supervisor.start(_step(tmp_path, "second"))

    supervisor.cleanup()
    supervisor.cleanup()

    assert first.terminated == second.terminated == 1
    assert first.killed == second.killed == 0
    with pytest.raises(SlurmRuntimeError, match="after cleanup"):
        supervisor.start(_step(tmp_path, "third"))


def test_wait_fails_when_required_service_exits(tmp_path: Path, fake_clock: FakeClock) -> None:
    service = _Process(1, returncode=19)
    target = _Process(2)
    supervisor = StepSupervisor(_Runner([service, target]), clock=fake_clock)
    managed_service = supervisor.start(_step(tmp_path, "service"))
    managed_target = supervisor.start(_step(tmp_path, "target"))

    with pytest.raises(SlurmRuntimeError, match="status 19"):
        supervisor.wait(managed_target, required=(managed_service,))


def test_subprocess_runner_uses_restrictive_logs_and_rejects_symlink_root(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir(mode=0o700)
    step = RuntimeStep(
        step_id="real-step",
        role=RuntimeStepRole.CLIENT_PREFLIGHT,
        command=(sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"),
        environment={"PATH": "/usr/bin", "LC_ALL": "C"},
        stdout_path=attempt / "logs" / "real-step.out",
        stderr_path=attempt / "logs" / "real-step.err",
    )

    process = SubprocessStepRunner().start(step)
    assert process.wait(timeout=10) == 0
    assert step.stdout_path.read_text() == "out\n"
    assert step.stderr_path.read_text() == "err\n"
    assert step.stdout_path.stat().st_mode & 0o077 == 0

    outside = tmp_path / "outside"
    outside.mkdir()
    symlink_attempt = tmp_path / "symlink-attempt"
    symlink_attempt.mkdir()
    (symlink_attempt / "logs").symlink_to(outside, target_is_directory=True)
    unsafe = RuntimeStep(
        step_id="unsafe-step",
        role=RuntimeStepRole.CLIENT,
        command=("true",),
        environment={"PATH": "/usr/bin"},
        stdout_path=symlink_attempt / "logs" / "unsafe.out",
        stderr_path=symlink_attempt / "logs" / "unsafe.err",
    )
    with pytest.raises(OSError, match="not a directory"):
        SubprocessStepRunner().start(unsafe)
    assert tuple(outside.iterdir()) == ()


def test_subprocess_cleanup_terminates_the_complete_step_process_group(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir(mode=0o700)
    parent_stopped = attempt / "parent-stopped"
    child_stopped = attempt / "child-stopped"
    child_ready = attempt / "child-ready"
    child_code = (
        "import signal,time; from pathlib import Path; "
        f"stopped=Path({child_stopped.as_posix()!r}); ready=Path({child_ready.as_posix()!r}); "
        "signal.signal(signal.SIGTERM, lambda *_: (stopped.write_text('stopped'), exit(0))); "
        "ready.write_text('ready'); time.sleep(60)"
    )
    parent_code = (
        "import signal,subprocess,sys,time; from pathlib import Path; "
        f"stopped=Path({parent_stopped.as_posix()!r}); child_code={child_code!r}; "
        "signal.signal(signal.SIGTERM, lambda *_: (stopped.write_text('stopped'), exit(0))); "
        "subprocess.Popen([sys.executable, '-c', child_code]); time.sleep(60)"
    )
    step = RuntimeStep(
        step_id="process-tree",
        role=RuntimeStepRole.SERVER,
        command=(sys.executable, "-c", parent_code),
        environment={"PATH": "/usr/bin"},
        stdout_path=attempt / "logs" / "process-tree.out",
        stderr_path=attempt / "logs" / "process-tree.err",
    )
    supervisor = StepSupervisor(
        SubprocessStepRunner(),
        poll_interval_seconds=0.01,
        termination_grace_seconds=3,
    )
    supervisor.start(step)
    deadline = time.monotonic() + 3
    while not child_ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert child_ready.exists()

    supervisor.cleanup()

    assert parent_stopped.read_text() == "stopped"
    assert child_stopped.read_text() == "stopped"


def test_start_normalizes_process_creation_failure(tmp_path: Path, fake_clock: FakeClock) -> None:
    class _FailingRunner:
        def start(self, step: RuntimeStep) -> _Process:
            del step
            raise subprocess.SubprocessError("injected")

    with pytest.raises(SlurmRuntimeError, match="cannot start"):
        StepSupervisor(_FailingRunner(), clock=fake_clock).start(_step(tmp_path, "failed"))


def _step(root: Path, step_id: str) -> RuntimeStep:
    return RuntimeStep(
        step_id=step_id,
        role=RuntimeStepRole.SERVER,
        command=("true",),
        environment={"PATH": "/usr/bin"},
        stdout_path=(root / f"{step_id}.out").absolute(),
        stderr_path=(root / f"{step_id}.err").absolute(),
    )
