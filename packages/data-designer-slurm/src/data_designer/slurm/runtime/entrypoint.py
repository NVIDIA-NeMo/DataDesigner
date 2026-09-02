# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allocation-local command entrypoint loaded from the checksummed runtime bundle."""

from __future__ import annotations

import argparse
import os
import signal
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Callable, Iterator

from data_designer.slurm.runtime.controller import OneNodeAllocationController
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.runtime.models import AllocationContext
from data_designer.slurm.runtime.preflight import SystemAllocationPreflight
from data_designer.slurm.runtime.probes import HttpReadinessProber
from data_designer.slurm.runtime.steps import DefaultClientStepBuilder
from data_designer.slurm.runtime.supervisor import StepSupervisor, SubprocessStepRunner, SystemRuntimeClock
from data_designer.slurm.state import SlurmStateWriter


def main(arguments: Sequence[str] | None = None) -> int:
    """Execute one allocation and return a bounded process status."""
    parser = argparse.ArgumentParser(prog="data-designer-slurm-runtime")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--attempt-dir", required=True)
    parsed = parser.parse_args(arguments)
    try:
        _run(Path(parsed.plan), Path(parsed.attempt_dir), os.environ)
    except SlurmRuntimeError as error:
        print(f"allocation runtime failed ({error.code.value}): {error}", file=sys.stderr)
        return (
            64 if error.code in {SlurmRuntimeErrorCode.INVALID_CONTEXT, SlurmRuntimeErrorCode.PREFLIGHT_FAILED} else 70
        )
    except KeyboardInterrupt:
        print("allocation runtime interrupted", file=sys.stderr)
        return 130
    except Exception:
        print("allocation runtime failed at an internal boundary", file=sys.stderr)
        return 70
    return 0


def _run(plan_path: Path, attempt_directory: Path, environment: Mapping[str, str]) -> None:
    if not plan_path.is_absolute() or not attempt_directory.is_absolute():
        raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime paths must be absolute")
    if plan_path.name != "resolved-plan.json" or plan_path.parent.parent.name != "runs":
        raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "resolved plan path is invalid")
    workspace_root = plan_path.parent.parent.parent
    run_id = plan_path.parent.name
    writer = SlurmStateWriter(workspace_root, run_id)
    plan = writer.load_resolved_plan()
    expected_plan_path = Path(plan.authored_config.path).with_name("resolved-plan.json")
    if plan_path != expected_plan_path:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.INVALID_CONTEXT,
            "runtime plan path does not match persisted run intent",
        )
    task_id = _scheduler_task_id(environment.get("SLURM_ARRAY_TASK_ID"))
    shards = tuple(shard for shard in plan.shards if shard.array_task_index == task_id)
    if len(shards) != 1:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.INVALID_CONTEXT,
            "scheduler array task does not identify exactly one planned shard",
        )
    shard = shards[0]
    expected_attempt_root = plan_path.parent / "shards" / shard.shard_id / "attempts"
    if attempt_directory.parent != expected_attempt_root:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.INVALID_CONTEXT,
            "attempt directory does not match the scheduler array task",
        )
    attempt = writer.load_attempt(shard.shard_id, attempt_directory.name)
    context = AllocationContext(
        plan=plan,
        shard=shard,
        attempt=attempt,
        attempt_directory=attempt_directory,
    )
    clock = SystemRuntimeClock()
    supervisor = StepSupervisor(SubprocessStepRunner(), clock=clock)
    controller = OneNodeAllocationController(
        context,
        runtime_proxy_path=Path(__file__).with_name("proxy.py"),
        state=writer,
        supervisor=supervisor,
        preflight=SystemAllocationPreflight(),
        client_steps=DefaultClientStepBuilder(),
        prober=HttpReadinessProber(),
        clock=clock,
        environment=environment,
    )
    with _interrupt_on_termination(supervisor.cleanup):
        controller.run()


def _scheduler_task_id(value: str | None) -> int:
    if value is None or not value.isascii() or not value.isdigit():
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.INVALID_CONTEXT,
            "SLURM_ARRAY_TASK_ID must be a non-negative integer",
        )
    return int(value)


@contextmanager
def _interrupt_on_termination(cleanup: Callable[[], None]) -> Iterator[None]:
    previous: dict[signal.Signals, signal.Handlers] = {}
    interrupted = False

    def interrupt(signum: int, frame: FrameType | None) -> None:
        nonlocal interrupted
        del signum, frame
        if interrupted:
            return
        interrupted = True
        try:
            cleanup()
        finally:
            raise KeyboardInterrupt

    for selected in (signal.SIGINT, signal.SIGTERM):
        previous[selected] = signal.getsignal(selected)
        signal.signal(selected, interrupt)
    try:
        yield
    finally:
        for selected, handler in previous.items():
            signal.signal(selected, handler)


if __name__ == "__main__":  # pragma: no cover - exercised through the installed module
    raise SystemExit(main())
