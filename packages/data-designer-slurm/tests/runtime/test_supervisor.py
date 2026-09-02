# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest
from slurm_test_fakes import FakeClock

from data_designer.slurm.runtime import signals as runtime_signals
from data_designer.slurm.runtime.errors import SlurmRuntimeError
from data_designer.slurm.runtime.models import RuntimeStep, RuntimeStepRole
from data_designer.slurm.runtime.signals import TerminationSignalCoordinator
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
    supervisor = StepSupervisor(runner, signals=TerminationSignalCoordinator(), clock=fake_clock)
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
    supervisor = StepSupervisor(
        _Runner([service, target]),
        signals=TerminationSignalCoordinator(),
        clock=fake_clock,
    )
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
        stdout_path=attempt / "logs" / "execution-00000001" / "real-step.out",
        stderr_path=attempt / "logs" / "execution-00000001" / "real-step.err",
    )

    process = SubprocessStepRunner().start(step)
    assert process.wait(timeout=10) == 0
    assert step.stdout_path.read_text() == "out\n"
    assert step.stderr_path.read_text() == "err\n"
    assert step.stdout_path.stat().st_mode & 0o077 == 0

    restarted = RuntimeStep(
        step_id=step.step_id,
        role=step.role,
        command=step.command,
        environment=step.environment,
        stdout_path=attempt / "logs" / "execution-00000002" / "real-step.out",
        stderr_path=attempt / "logs" / "execution-00000002" / "real-step.err",
    )
    restarted_process = SubprocessStepRunner().start(restarted)
    assert restarted_process.wait(timeout=10) == 0
    assert restarted.stdout_path.read_text() == "out\n"

    outside = tmp_path / "outside"
    outside.mkdir()
    symlink_attempt = tmp_path / "symlink-attempt"
    symlink_attempt.mkdir(mode=0o700)
    (symlink_attempt / "logs").symlink_to(outside, target_is_directory=True)
    unsafe = RuntimeStep(
        step_id="unsafe-step",
        role=RuntimeStepRole.CLIENT,
        command=("true",),
        environment={"PATH": "/usr/bin"},
        stdout_path=symlink_attempt / "logs" / "execution-00000001" / "unsafe.out",
        stderr_path=symlink_attempt / "logs" / "execution-00000001" / "unsafe.err",
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
        stdout_path=attempt / "logs" / "execution-00000001" / "process-tree.out",
        stderr_path=attempt / "logs" / "execution-00000001" / "process-tree.err",
    )
    supervisor = StepSupervisor(
        SubprocessStepRunner(),
        signals=TerminationSignalCoordinator(),
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
        StepSupervisor(
            _FailingRunner(),
            signals=TerminationSignalCoordinator(),
            clock=fake_clock,
        ).start(_step(tmp_path, "failed"))


def test_start_registers_process_before_replaying_deferred_termination(
    tmp_path: Path,
    fake_clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _Process(1)
    installed_handlers: dict[object, object] = {}
    signals = TerminationSignalCoordinator()
    supervisor: StepSupervisor

    def fake_getsignal(selected: object) -> object:
        del selected
        return runtime_signals.signal.SIG_DFL

    def fake_signal(selected: object, handler: object) -> object:
        installed_handlers[selected] = handler
        return runtime_signals.signal.SIG_DFL

    def fake_kill(process_id: int, selected: object) -> None:
        del process_id
        assert len(supervisor.managed_steps) == 1
        handler = installed_handlers[selected]
        assert callable(handler)
        handler(selected, None)

    class _InterruptingRunner(_Runner):
        def start(self, step: RuntimeStep) -> _Process:
            started = super().start(step)
            handler = installed_handlers[runtime_signals.signal.SIGTERM]
            assert callable(handler)
            handler(runtime_signals.signal.SIGTERM, None)
            return started

    monkeypatch.setattr(runtime_signals.signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(runtime_signals.signal, "signal", fake_signal)
    monkeypatch.setattr(runtime_signals.os, "kill", fake_kill)
    supervisor = StepSupervisor(_InterruptingRunner([process]), signals=signals, clock=fake_clock)

    with (
        pytest.raises(KeyboardInterrupt),
        signals.interrupt_on_termination(supervisor.cleanup),
    ):
        supervisor.start(_step(tmp_path, "server"))

    assert supervisor.managed_steps[0].process is process
    assert process.terminated == 1


def test_start_can_register_from_a_worker_thread(tmp_path: Path, fake_clock: FakeClock) -> None:
    signals = TerminationSignalCoordinator()
    supervisor = StepSupervisor(_Runner([_Process(1)]), signals=signals, clock=fake_clock)

    with (
        signals.interrupt_on_termination(supervisor.cleanup),
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        managed = executor.submit(supervisor.start, _step(tmp_path, "server")).result()

    assert supervisor.managed_steps == (managed,)
    supervisor.cleanup()


def test_cleanup_remains_incomplete_while_a_process_is_still_running(
    tmp_path: Path,
    fake_clock: FakeClock,
) -> None:
    class _StubbornProcess(_Process):
        def terminate(self) -> None:
            self.terminated += 1

        def kill(self) -> None:
            self.killed += 1

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            raise TimeoutError("still running")

    process = _StubbornProcess(1)
    supervisor = StepSupervisor(
        _Runner([process]),
        signals=TerminationSignalCoordinator(),
        clock=fake_clock,
        poll_interval_seconds=1,
        termination_grace_seconds=1,
    )
    supervisor.start(_step(tmp_path, "server"))

    with pytest.raises(SlurmRuntimeError, match="cleanup failed"):
        supervisor.cleanup()

    assert process.terminated == process.killed == 1
    assert not supervisor.cleanup_complete


def _step(root: Path, step_id: str) -> RuntimeStep:
    return RuntimeStep(
        step_id=step_id,
        role=RuntimeStepRole.SERVER,
        command=("true",),
        environment={"PATH": "/usr/bin"},
        stdout_path=(root / f"{step_id}.out").absolute(),
        stderr_path=(root / f"{step_id}.err").absolute(),
    )
