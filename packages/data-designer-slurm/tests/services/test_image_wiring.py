# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pytest
from slurm_test_fakes import FakeClock

from data_designer.slurm.config import (
    ClientImageInspection,
    ImageBuildProfile,
    ImageBuildRequest,
    ImageInspectionRecord,
    ImageKind,
    ImageRef,
    InstalledDistribution,
    SchedulerProfile,
    SlurmProfile,
)
from data_designer.slurm.images.service import VerifiedImageRegistry
from data_designer.slurm.launcher.errors import SlurmSubmissionError
from data_designer.slurm.launcher.models import (
    SlurmAccountingEntry,
    SlurmJobSubmissionReceipt,
    SlurmProcessExitCode,
    SlurmQueueEntry,
)
from data_designer.slurm.services import (
    SlurmImageService,
    SlurmServiceError,
    SlurmServiceErrorCode,
    create_slurm_image_service,
)
from data_designer.slurm.state import SchedulerState

_JOB_ID = 42
_LIFECYCLE_ID = "image-test"


class _Launcher:
    def __init__(
        self,
        *,
        queue_state: SchedulerState | None = None,
        accounting_state: SchedulerState | None = SchedulerState.COMPLETED,
        exit_status: int = 0,
        termination_signal: int = 0,
        on_submit: Callable[[], None] | None = None,
        submission_error: SlurmSubmissionError | None = None,
    ) -> None:
        self.queue_state = queue_state
        self.accounting_state = accounting_state
        self.exit_status = exit_status
        self.termination_signal = termination_signal
        self.on_submit = on_submit
        self.submission_error = submission_error
        self.submissions: list[str] = []
        self.cancellations: list[int] = []

    def submit_script(self, script: str, **_: object) -> SlurmJobSubmissionReceipt:
        self.submissions.append(script)
        if self.submission_error is not None:
            raise self.submission_error
        if self.on_submit is not None:
            self.on_submit()
        return SlurmJobSubmissionReceipt(job_id=_JOB_ID)

    def query_queue(self, selectors: object) -> tuple[SlurmQueueEntry, ...]:
        del selectors
        if self.queue_state is None:
            return ()
        return (SlurmQueueEntry(job_identity=_JOB_ID, state=self.queue_state),)

    def query_accounting(self, selectors: object) -> tuple[SlurmAccountingEntry, ...]:
        del selectors
        if self.accounting_state is None:
            return ()
        return (
            SlurmAccountingEntry(
                job_identity=_JOB_ID,
                state=self.accounting_state,
                process_exit_code=SlurmProcessExitCode(
                    exit_status=self.exit_status,
                    termination_signal=self.termination_signal,
                ),
            ),
        )

    def cancel(self, job_id: int) -> None:
        self.cancellations.append(job_id)


class _DelayedAccountingLauncher(_Launcher):
    def __init__(self, *, on_submit: Callable[[], None]) -> None:
        super().__init__(queue_state=SchedulerState.COMPLETED, on_submit=on_submit)
        self.queue_queries = 0
        self.accounting_queries = 0

    def query_queue(self, selectors: object) -> tuple[SlurmQueueEntry, ...]:
        self.queue_queries += 1
        if self.queue_queries == 1:
            return super().query_queue(selectors)
        return ()

    def query_accounting(self, selectors: object) -> tuple[SlurmAccountingEntry, ...]:
        self.accounting_queries += 1
        if self.accounting_queries < 3:
            return (
                SlurmAccountingEntry(
                    job_identity=_JOB_ID,
                    state=SchedulerState.RUNNING,
                    process_exit_code=SlurmProcessExitCode(exit_status=0, termination_signal=0),
                ),
            )
        return super().query_accounting(selectors)


class _DelayedCancellationLauncher(_Launcher):
    def __init__(self) -> None:
        super().__init__(accounting_state=None)

    def cancel(self, job_id: int) -> None:
        super().cancel(job_id)
        self.accounting_state = SchedulerState.RUNNING

    def query_accounting(self, selectors: object) -> tuple[SlurmAccountingEntry, ...]:
        entries = super().query_accounting(selectors)
        if self.cancellations:
            self.accounting_state = SchedulerState.CANCELLED
        return entries


class _ImmediateCancellationLauncher(_Launcher):
    def cancel(self, job_id: int) -> None:
        super().cancel(job_id)
        self.accounting_state = SchedulerState.CANCELLED


def _interrupt(_: float) -> None:
    raise KeyboardInterrupt


def test_default_image_add_runs_lifecycle_and_registers_existing_sqsh(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    source = tmp_path / "client.sqsh"
    content = b"client image"
    source.write_bytes(content)
    launcher = _Launcher(on_submit=lambda: _write_inspection(workspace, content))
    service = _service(workspace, launcher)

    registered = service.add(ImageBuildRequest(name="client", kind="client", source=source.as_posix()))

    assert registered.path == source.as_posix()
    assert registered.sqsh_sha256 == hashlib.sha256(content).hexdigest()
    assert service.get("client") == registered
    assert (
        VerifiedImageRegistry(workspace)
        .resolve_for_planning(
            ImageRef(name="client"),
            expected_kind=ImageKind.CLIENT,
        )
        .path
        == source.as_posix()
    )
    assert len(launcher.submissions) == 1
    assert "#SBATCH --partition=cpu" in launcher.submissions[0]
    assert not _job_directory(workspace).exists()


def test_default_image_add_waits_for_completed_job_accounting_evidence(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    source = tmp_path / "client.sqsh"
    content = b"client image"
    source.write_bytes(content)
    launcher = _DelayedAccountingLauncher(on_submit=lambda: _write_inspection(workspace, content))
    clock = FakeClock(datetime(2026, 9, 10, tzinfo=timezone.utc))

    registered = _service(workspace, launcher, clock=clock).add(
        ImageBuildRequest(name="client", kind="client", source=source.as_posix())
    )

    assert registered.path == source.as_posix()
    assert launcher.queue_queries == 3
    assert launcher.accounting_queries == 3
    assert clock.sleep_calls == [300.0, 300.0]
    assert not _job_directory(workspace).exists()


def test_default_image_add_preserves_collision_then_replaces_explicitly(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    first = tmp_path / "first.sqsh"
    second = tmp_path / "second.sqsh"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    current_content = b"first"
    launcher = _Launcher(on_submit=lambda: _write_inspection(workspace, current_content))
    service = _service(workspace, launcher)
    original = service.add(ImageBuildRequest(name="client", kind="client", source=first.as_posix()))
    current_content = b"second"

    with pytest.raises(SlurmServiceError) as caught:
        service.add(ImageBuildRequest(name="client", kind="client", source=second.as_posix()))

    assert caught.value.code is SlurmServiceErrorCode.CONFLICT
    assert service.get("client") == original

    replacement = service.add(
        ImageBuildRequest(name="client", kind="client", source=second.as_posix()),
        replace=True,
    )

    assert replacement.path == second.as_posix()
    assert service.get("client") == replacement
    assert not _job_directory(workspace).exists()


@pytest.mark.parametrize(
    ("state", "exit_status"),
    ((SchedulerState.FAILED, 1), (SchedulerState.COMPLETED, 1)),
    ids=("terminal-failure", "nonzero-exit"),
)
def test_default_image_add_fails_closed_and_cleans_terminal_jobs(
    tmp_path: Path,
    state: SchedulerState,
    exit_status: int,
) -> None:
    workspace = tmp_path / "workspace"
    source = tmp_path / "client.sqsh"
    source.write_bytes(b"client")
    launcher = _Launcher(accounting_state=state, exit_status=exit_status)

    with pytest.raises(SlurmServiceError) as caught:
        _service(workspace, launcher).add(ImageBuildRequest(name="client", kind="client", source=source.as_posix()))

    assert caught.value.code is SlurmServiceErrorCode.UNAVAILABLE
    assert not _job_directory(workspace).exists()
    assert not (workspace / "images" / "registry.yaml").exists()


def test_default_image_add_cleans_definitive_submission_failure(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    source = tmp_path / "client.sqsh"
    source.write_bytes(b"client")
    launcher = _Launcher(
        submission_error=SlurmSubmissionError("rejected", may_have_succeeded=False),
    )

    with pytest.raises(SlurmServiceError) as caught:
        _service(workspace, launcher).add(ImageBuildRequest(name="client", kind="client", source=source.as_posix()))

    assert caught.value.code is SlurmServiceErrorCode.UNAVAILABLE
    assert not _job_directory(workspace).exists()


def test_default_image_add_retains_ambiguous_submission_state(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    source = tmp_path / "client.sqsh"
    source.write_bytes(b"client")
    launcher = _Launcher(
        submission_error=SlurmSubmissionError("unknown", may_have_succeeded=True),
    )

    with pytest.raises(SlurmServiceError) as caught:
        _service(workspace, launcher).add(ImageBuildRequest(name="client", kind="client", source=source.as_posix()))

    assert caught.value.code is SlurmServiceErrorCode.UNAVAILABLE
    assert str(caught.value) == ("image lifecycle submission outcome is unknown; lifecycle image-test was retained")
    assert _job_directory(workspace).is_dir()


def test_default_image_add_waits_for_cancelled_job_before_cleanup(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    source = tmp_path / "client.sqsh"
    source.write_bytes(b"client")
    launcher = _DelayedCancellationLauncher()
    clock = FakeClock(datetime(2026, 9, 10, tzinfo=timezone.utc))

    with pytest.raises(SlurmServiceError) as caught:
        _service(workspace, launcher, clock=clock).add(
            ImageBuildRequest(name="client", kind="client", source=source.as_posix())
        )

    assert caught.value.code is SlurmServiceErrorCode.UNAVAILABLE
    assert launcher.cancellations == [_JOB_ID]
    assert clock.sleep_calls == [300.0, 300.0, 300.0]
    assert not _job_directory(workspace).exists()


def test_default_image_add_retains_state_without_terminal_cancellation_evidence(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    source = tmp_path / "client.sqsh"
    source.write_bytes(b"client")
    launcher = _Launcher(accounting_state=None)
    clock = FakeClock(datetime(2026, 9, 10, tzinfo=timezone.utc))

    with pytest.raises(SlurmServiceError):
        _service(workspace, launcher, clock=clock).add(
            ImageBuildRequest(name="client", kind="client", source=source.as_posix())
        )

    assert launcher.cancellations == [_JOB_ID]
    assert clock.sleep_calls == [300.0, 300.0, 300.0]
    assert _job_directory(workspace).is_dir()


def test_default_image_add_retains_completed_job_without_accounting_exit_evidence(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    source = tmp_path / "client.sqsh"
    source.write_bytes(b"client")
    launcher = _Launcher(queue_state=SchedulerState.COMPLETED, accounting_state=None)
    clock = FakeClock(datetime(2026, 9, 10, tzinfo=timezone.utc))

    with pytest.raises(SlurmServiceError) as caught:
        _service(workspace, launcher, clock=clock).add(
            ImageBuildRequest(name="client", kind="client", source=source.as_posix())
        )

    assert caught.value.code is SlurmServiceErrorCode.UNAVAILABLE
    assert launcher.cancellations == [_JOB_ID]
    assert clock.sleep_calls == [300.0, 300.0, 300.0]
    assert _job_directory(workspace).is_dir()


def test_default_image_add_cancels_and_cleans_on_interrupt(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    source = tmp_path / "client.sqsh"
    source.write_bytes(b"client")
    launcher = _ImmediateCancellationLauncher(queue_state=SchedulerState.PENDING, accounting_state=None)

    service = create_slurm_image_service(
        profile=_profile(workspace),
        launcher=launcher,  # type: ignore[arg-type]
        lifecycle_id_factory=lambda: _LIFECYCLE_ID,
        sleep=_interrupt,
    )

    with pytest.raises(KeyboardInterrupt):
        service.add(ImageBuildRequest(name="client", kind="client", source=source.as_posix()))

    assert launcher.cancellations == [_JOB_ID]
    assert not _job_directory(workspace).exists()


def _service(workspace: Path, launcher: _Launcher, *, clock: FakeClock | None = None) -> SlurmImageService:
    return create_slurm_image_service(
        profile=_profile(workspace),
        launcher=launcher,  # type: ignore[arg-type]
        lifecycle_id_factory=lambda: _LIFECYCLE_ID,
        clock=None if clock is None else clock.now,
        sleep=None if clock is None else clock.sleep,
    )


def _profile(workspace: Path) -> SlurmProfile:
    return SlurmProfile(
        schema_version=1,
        scheduler=SchedulerProfile(account="research", partition="gpu"),
        gpus_per_node=8,
        workspace_root=workspace.as_posix(),
        image_build=ImageBuildProfile(
            partition="cpu",
            cpus_per_task=2,
            memory="8G",
            time_limit="04:00:00",
        ),
    )


def _write_inspection(workspace: Path, content: bytes) -> None:
    inspection = ImageInspectionRecord(
        schema_version=1,
        inspector_version="inspector-1",
        sqsh_sha256=hashlib.sha256(content).hexdigest(),
        inspection=ClientImageInspection(
            kind="client",
            python_implementation="cpython",
            python_version="3.13.3",
            python_abi="cp313",
            distributions=tuple(
                InstalledDistribution(name=name, version="0.9.2")
                for name in (
                    "data-designer",
                    "data-designer-config",
                    "data-designer-engine",
                    "data-designer-slurm",
                )
            )
            + (InstalledDistribution(name="pip", version="26.1"),),
            installer_path="/usr/bin/pip",
            installer_version="26.1",
        ),
    )
    output = _job_directory(workspace) / "output" / "inspection.json"
    output.write_text(inspection.model_dump_json())


def _job_directory(workspace: Path) -> Path:
    return workspace / "images" / ".tmp" / "jobs" / _LIFECYCLE_ID
