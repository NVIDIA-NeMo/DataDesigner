# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable

import pytest
from conftest import FakeClientStepBuilder, FakePreflight, FakeStateStore, RuntimeCase
from slurm_test_fakes import FakeClock

from data_designer.slurm.client import ClientOutcome, ClientResult
from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.runtime.controller import OneNodeAllocationController
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.runtime.models import RuntimeStep, RuntimeStepRole
from data_designer.slurm.runtime.supervisor import StepSupervisor
from data_designer.slurm.state import (
    AttemptLifecycleState,
    AttemptTerminalClassification,
    CandidateOutcome,
    CandidateOutputFile,
    CandidateOutputManifest,
    EndpointPublicationState,
    ReadinessState,
)


@dataclass(slots=True)
class _FakeProcess:
    pid: int
    returncode: int | None
    fail_terminate: bool = False
    terminate_calls: int = 0
    kill_calls: int = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.fail_terminate:
            raise OSError("injected termination failure")
        self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            raise TimeoutError
        return self.returncode


class _FakeRunner:
    def __init__(
        self,
        *,
        generation_hook: Callable[[], None] | None = None,
        server_returncode: int | None = None,
        failed_role: RuntimeStepRole | None = None,
        fail_cleanup: bool = False,
    ) -> None:
        self.generation_hook = generation_hook
        self.server_returncode = server_returncode
        self.failed_role = failed_role
        self.fail_cleanup = fail_cleanup
        self.steps: list[RuntimeStep] = []
        self.processes: list[_FakeProcess] = []

    def start(self, step: RuntimeStep) -> _FakeProcess:
        self.steps.append(step)
        if step.role is RuntimeStepRole.CLIENT and self.generation_hook is not None:
            self.generation_hook()
        is_managed_service = step.role in {RuntimeStepRole.SERVER, RuntimeStepRole.ENDPOINT}
        if step.role is self.failed_role:
            returncode = 23
        elif step.role is RuntimeStepRole.SERVER:
            returncode = self.server_returncode
        else:
            returncode = None if is_managed_service else 0
        process = _FakeProcess(
            pid=1000 + len(self.processes),
            returncode=returncode,
            fail_terminate=self.fail_cleanup and is_managed_service,
        )
        self.processes.append(process)
        return process


class _FakeProber:
    def __init__(self, *, ready: bool, clock: FakeClock, advance_on_failure: float = 0) -> None:
        self.ready = ready
        self.clock = clock
        self.advance_on_failure = advance_on_failure
        self.calls = 0

    def is_ready(self, host: str, port: int, path: str, *, timeout_seconds: float) -> bool:
        del host, port, path, timeout_seconds
        self.calls += 1
        if not self.ready and self.advance_on_failure:
            self.clock.advance(self.advance_on_failure)
        return self.ready


def test_controller_runs_preflight_servers_endpoint_client_and_cleanup(runtime_case: RuntimeCase) -> None:
    clock = FakeClock(runtime_case.created_at.replace(second=10), monotonic_time=100)
    state = FakeStateStore(runtime_case.context.attempt)
    runner = _FakeRunner(generation_hook=lambda: _write_complete_result(runtime_case, clock))
    supervisor = StepSupervisor(runner, clock=clock, poll_interval_seconds=0.1)
    controller = OneNodeAllocationController(
        runtime_case.context,
        runtime_proxy_path=runtime_case.context.attempt_directory / "runtime/proxy.py",
        state=state,
        supervisor=supervisor,
        preflight=FakePreflight(),
        client_steps=FakeClientStepBuilder(),
        prober=_FakeProber(ready=True, clock=clock),
        clock=clock,
        environment={},
    )

    result = controller.run()

    assert result.state is AttemptLifecycleState.SUCCEEDED
    assert result.terminal_classification is AttemptTerminalClassification.SUCCEEDED
    assert result.candidate_output is not None
    assert [step.role for step in runner.steps] == [
        RuntimeStepRole.CLIENT_PREFLIGHT,
        RuntimeStepRole.SERVER,
        RuntimeStepRole.ENDPOINT,
        RuntimeStepRole.CLIENT,
    ]
    assert [item.state for item in state.readiness][0] is ReadinessState.PENDING
    assert ReadinessState.READY in [item.state for item in state.readiness]
    assert state.readiness[-1].state is ReadinessState.STOPPED
    assert state.readiness[-1].deployments[0].endpoint_publication is EndpointPublicationState.PUBLISHED
    assert all(process.poll() is not None for process in runner.processes)


def test_preflight_failure_starts_no_process_and_fails_attempt(runtime_case: RuntimeCase) -> None:
    clock = FakeClock(runtime_case.created_at.replace(second=10), monotonic_time=100)
    state = FakeStateStore(runtime_case.context.attempt)
    runner = _FakeRunner()
    failure = SlurmRuntimeError(SlurmRuntimeErrorCode.PREFLIGHT_FAILED, "injected preflight failure")
    controller = OneNodeAllocationController(
        runtime_case.context,
        runtime_proxy_path=runtime_case.context.attempt_directory / "runtime/proxy.py",
        state=state,
        supervisor=StepSupervisor(runner, clock=clock),
        preflight=FakePreflight(failure),
        client_steps=FakeClientStepBuilder(),
        prober=_FakeProber(ready=True, clock=clock),
        clock=clock,
        environment={},
    )

    with pytest.raises(SlurmRuntimeError, match="injected preflight failure") as raised:
        controller.run()

    assert raised.value.__cause__ is None
    assert runner.steps == []
    assert state.readiness == []
    assert state.attempt.state is AttemptLifecycleState.FAILED


def test_required_server_exit_fails_and_cleans_partial_start(runtime_case: RuntimeCase) -> None:
    clock = FakeClock(runtime_case.created_at.replace(second=10), monotonic_time=100)
    state = FakeStateStore(runtime_case.context.attempt)
    runner = _FakeRunner(server_returncode=17)
    controller = OneNodeAllocationController(
        runtime_case.context,
        runtime_proxy_path=runtime_case.context.attempt_directory / "runtime/proxy.py",
        state=state,
        supervisor=StepSupervisor(runner, clock=clock),
        preflight=FakePreflight(),
        client_steps=FakeClientStepBuilder(),
        prober=_FakeProber(ready=True, clock=clock),
        clock=clock,
        environment={},
    )

    with pytest.raises(SlurmRuntimeError, match="required runtime step"):
        controller.run()

    assert state.attempt.state is AttemptLifecycleState.FAILED
    assert ReadinessState.FAILED in [item.state for item in state.readiness]
    assert state.readiness[-1].state is ReadinessState.STOPPED


def test_readiness_timeout_fails_and_terminates_server(runtime_case: RuntimeCase) -> None:
    clock = FakeClock(runtime_case.created_at.replace(second=10), monotonic_time=100)
    state = FakeStateStore(runtime_case.context.attempt)
    runner = _FakeRunner()
    controller = OneNodeAllocationController(
        runtime_case.context,
        runtime_proxy_path=runtime_case.context.attempt_directory / "runtime/proxy.py",
        state=state,
        supervisor=StepSupervisor(runner, clock=clock),
        preflight=FakePreflight(),
        client_steps=FakeClientStepBuilder(),
        prober=_FakeProber(ready=False, clock=clock, advance_on_failure=10_000),
        clock=clock,
        environment={},
    )

    with pytest.raises(SlurmRuntimeError, match="readiness timed out"):
        controller.run()

    server = next(
        process
        for step, process in zip(runner.steps, runner.processes, strict=True)
        if step.role is RuntimeStepRole.SERVER
    )
    assert server.terminate_calls == 1
    assert server.poll() == -15
    assert state.attempt.state is AttemptLifecycleState.FAILED


@pytest.mark.parametrize(
    "failed_role",
    (RuntimeStepRole.CLIENT_PREFLIGHT, RuntimeStepRole.ENDPOINT, RuntimeStepRole.CLIENT),
)
def test_managed_step_failure_fails_attempt_and_cleans_started_services(
    runtime_case: RuntimeCase,
    failed_role: RuntimeStepRole,
) -> None:
    clock = FakeClock(runtime_case.created_at.replace(second=10), monotonic_time=100)
    state = FakeStateStore(runtime_case.context.attempt)
    runner = _FakeRunner(failed_role=failed_role)
    controller = OneNodeAllocationController(
        runtime_case.context,
        runtime_proxy_path=runtime_case.context.attempt_directory / "runtime/proxy.py",
        state=state,
        supervisor=StepSupervisor(runner, clock=clock),
        preflight=FakePreflight(),
        client_steps=FakeClientStepBuilder(),
        prober=_FakeProber(ready=True, clock=clock),
        clock=clock,
        environment={},
    )

    with pytest.raises(SlurmRuntimeError, match="status 23"):
        controller.run()

    assert state.attempt.state is AttemptLifecycleState.FAILED
    assert state.readiness[-1].state is ReadinessState.STOPPED
    services = (
        process
        for step, process in zip(runner.steps, runner.processes, strict=True)
        if step.role in {RuntimeStepRole.SERVER, RuntimeStepRole.ENDPOINT}
    )
    assert all(process.poll() is not None for process in services)


def test_cleanup_failure_prevents_false_success(runtime_case: RuntimeCase) -> None:
    clock = FakeClock(runtime_case.created_at.replace(second=10), monotonic_time=100)
    state = FakeStateStore(runtime_case.context.attempt)
    runner = _FakeRunner(
        generation_hook=lambda: _write_complete_result(runtime_case, clock),
        fail_cleanup=True,
    )
    controller = OneNodeAllocationController(
        runtime_case.context,
        runtime_proxy_path=runtime_case.context.attempt_directory / "runtime/proxy.py",
        state=state,
        supervisor=StepSupervisor(
            runner,
            clock=clock,
            poll_interval_seconds=1,
            termination_grace_seconds=1,
        ),
        preflight=FakePreflight(),
        client_steps=FakeClientStepBuilder(),
        prober=_FakeProber(ready=True, clock=clock),
        clock=clock,
        environment={},
    )

    with pytest.raises(SlurmRuntimeError, match="cleanup failed"):
        controller.run()

    assert state.attempt.state is AttemptLifecycleState.FAILED
    assert all(process.poll() is not None for process in runner.processes)


def test_cleanup_failure_is_retained_as_a_note_on_the_primary_failure(runtime_case: RuntimeCase) -> None:
    clock = FakeClock(runtime_case.created_at.replace(second=10), monotonic_time=100)
    state = FakeStateStore(runtime_case.context.attempt)
    runner = _FakeRunner(failed_role=RuntimeStepRole.CLIENT, fail_cleanup=True)
    controller = OneNodeAllocationController(
        runtime_case.context,
        runtime_proxy_path=runtime_case.context.attempt_directory / "runtime/proxy.py",
        state=state,
        supervisor=StepSupervisor(
            runner,
            clock=clock,
            poll_interval_seconds=1,
            termination_grace_seconds=1,
        ),
        preflight=FakePreflight(),
        client_steps=FakeClientStepBuilder(),
        prober=_FakeProber(ready=True, clock=clock),
        clock=clock,
        environment={},
    )

    with pytest.raises(SlurmRuntimeError, match="status 23") as raised:
        controller.run()

    assert any("cleanup also failed" in note for note in raised.value.__notes__)
    assert state.attempt.state is AttemptLifecycleState.FAILED


def test_future_client_timestamp_cannot_push_persisted_state_clock_forward(runtime_case: RuntimeCase) -> None:
    clock = FakeClock(runtime_case.created_at.replace(second=10), monotonic_time=100)
    state = FakeStateStore(runtime_case.context.attempt)

    def write_future_result() -> None:
        _write_complete_result(runtime_case, clock)
        clock.current_time -= timedelta(milliseconds=500)

    runner = _FakeRunner(generation_hook=write_future_result)
    controller = OneNodeAllocationController(
        runtime_case.context,
        runtime_proxy_path=runtime_case.context.attempt_directory / "runtime/proxy.py",
        state=state,
        supervisor=StepSupervisor(runner, clock=clock),
        preflight=FakePreflight(),
        client_steps=FakeClientStepBuilder(),
        prober=_FakeProber(ready=True, clock=clock),
        clock=clock,
        environment={},
    )

    with pytest.raises(SlurmRuntimeError, match="later than the allocation clock"):
        controller.run()

    assert state.attempt.state is AttemptLifecycleState.FAILED


def test_stale_candidate_from_an_earlier_generation_cannot_succeed(runtime_case: RuntimeCase) -> None:
    clock = FakeClock(runtime_case.created_at.replace(second=10), monotonic_time=100)
    _write_complete_result(runtime_case, clock)
    clock.advance(5)
    state = FakeStateStore(runtime_case.context.attempt)
    controller = OneNodeAllocationController(
        runtime_case.context,
        runtime_proxy_path=runtime_case.context.attempt_directory / "runtime/proxy.py",
        state=state,
        supervisor=StepSupervisor(_FakeRunner(), clock=clock),
        preflight=FakePreflight(),
        client_steps=FakeClientStepBuilder(),
        prober=_FakeProber(ready=True, clock=clock),
        clock=clock,
        environment={},
    )

    with pytest.raises(SlurmRuntimeError, match="predates the current generation"):
        controller.run()

    assert state.attempt.state is AttemptLifecycleState.FAILED


def _write_complete_result(runtime_case: RuntimeCase, clock: FakeClock) -> None:
    context = runtime_case.context
    requested = context.shard.requested_records
    dataset_path = (context.attempt_directory / "dataset").as_posix()
    candidate = CandidateOutputManifest(
        schema_version=1,
        run_id=context.plan.run_id,
        shard_id=context.shard.shard_id,
        attempt_id=context.attempt.attempt_id,
        attempt_ordinal=context.attempt.attempt_ordinal,
        created_at=clock.now(),
        dataset_path=dataset_path,
        requested_records=requested,
        actual_records=requested,
        outcome=CandidateOutcome.COMPLETE,
        files=(
            CandidateOutputFile(
                relative_path="part-00000.parquet",
                sha256="a" * 64,
                byte_size=128,
                record_count=requested,
            ),
        ),
        dataset_schema_digest="b" * 64,
        provenance_digest="c" * 64,
    )
    candidate_path = context.attempt_directory / "output-manifest.json"
    _write_record(candidate_path, candidate.serialize_json())
    clock.advance(1)
    result = ClientResult(
        schema_version=1,
        run_id=context.plan.run_id,
        shard_id=context.shard.shard_id,
        attempt_id=context.attempt.attempt_id,
        completed_at=clock.now(),
        requested_records=requested,
        actual_records=requested,
        outcome=ClientOutcome.COMPLETE,
        dataset_path=dataset_path,
        early_shutdown=False,
        requested_resume_mode=context.plan.invocation.authored.resume,
        effective_resume_mode="never",
        candidate_output_manifest=ArtifactReference(
            path=candidate_path.as_posix(),
            sha256=candidate.compute_sha256(),
        ),
    )
    _write_record(context.attempt_directory / "client-result.json", result.serialize_json())


def _write_record(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o600)
