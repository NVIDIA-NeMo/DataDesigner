# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

from data_designer.slurm.runtime import node_worker as runtime_node_worker
from data_designer.slurm.runtime.node_spec import (
    NodeProcessSpec,
    NodeSpec,
    NodeWorkerSpec,
    decode_node_worker_spec,
    encode_node_worker_spec,
)
from data_designer.slurm.runtime.node_worker import NodeProcessSupervisor


def test_node_worker_spec_round_trips_and_rejects_untrusted_payload() -> None:
    spec = _worker_spec((NodeProcessSpec("lane-0", ("true",), (0,), 0),))
    assert decode_node_worker_spec(encode_node_worker_spec(spec)) == spec
    with pytest.raises(ValueError, match="invalid"):
        decode_node_worker_spec("not-base64")


def test_partial_startup_failure_cleans_already_started_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess(pid=41)
    calls = 0
    signals: list[int] = []

    def start(*arguments: object, **keywords: object) -> _FakeProcess:
        nonlocal calls
        del arguments, keywords
        calls += 1
        if calls == 2:
            raise OSError("injected partial startup")
        return process

    monkeypatch.setattr(runtime_node_worker.subprocess, "Popen", start)

    def terminate(child: _FakeProcess, selected: int) -> None:
        signals.append(selected)
        child.returncode = -15

    monkeypatch.setattr(runtime_node_worker, "_signal_process_group", terminate)
    supervisor = NodeProcessSupervisor(environment={})
    node = NodeSpec(
        node_index=0,
        host="compute-001",
        ports=(),
        processes=(
            NodeProcessSpec("lane-0", ("first",), (0,), 0),
            NodeProcessSpec("lane-1", ("second",), (1,), 0),
        ),
    )

    with pytest.raises(OSError, match="partial startup"):
        supervisor.run(node, ("0", "1"))

    assert signals
    assert supervisor.cleanup_complete


def test_follower_failure_terminates_sibling_without_orphan(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    stopped = tmp_path / "stopped"
    long_running = (
        "import signal,time; from pathlib import Path; "
        f"ready=Path({ready.as_posix()!r}); stopped=Path({stopped.as_posix()!r}); "
        "signal.signal(signal.SIGTERM, lambda *_: (stopped.write_text('stopped'), exit(0))); "
        "ready.write_text('ready'); time.sleep(30)"
    )
    failing = "import time; time.sleep(0.2); raise SystemExit(23)"
    node = NodeSpec(
        node_index=0,
        host="compute-001",
        ports=(),
        processes=(
            NodeProcessSpec("head", (sys.executable, "-c", long_running), (0,), 0),
            NodeProcessSpec("follower", (sys.executable, "-c", failing), (1,), 0),
        ),
    )
    supervisor = NodeProcessSupervisor(environment={})

    assert supervisor.run(node, ("0", "1")) == 23
    supervisor.cleanup()

    assert ready.exists()
    assert stopped.read_text() == "stopped"
    assert supervisor.cleanup_complete


def test_cancellation_request_stops_all_node_lanes(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    stopped = tmp_path / "stopped"
    child = (
        "import signal,time; from pathlib import Path; "
        f"ready=Path({ready.as_posix()!r}); stopped=Path({stopped.as_posix()!r}); "
        "signal.signal(signal.SIGTERM, lambda *_: (stopped.write_text('stopped'), exit(0))); "
        "ready.write_text('ready'); time.sleep(30)"
    )
    node = NodeSpec(
        node_index=0,
        host="compute-001",
        ports=(),
        processes=(NodeProcessSpec("lane", (sys.executable, "-c", child), (0,), 0),),
    )
    supervisor = NodeProcessSupervisor(environment={})
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(supervisor.run, node, ("0",))
        _wait_for_file(ready)
        supervisor.request_stop()
        assert result.result(timeout=5) == 1

    assert stopped.read_text() == "stopped"
    assert supervisor.cleanup_complete


def test_cancellation_interrupts_a_pending_launch_delay() -> None:
    node = NodeSpec(
        node_index=0,
        host="compute-001",
        ports=(),
        processes=(NodeProcessSpec("delayed", ("must-not-start",), (0,), 30),),
    )
    supervisor = NodeProcessSupervisor(environment={})
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(supervisor.run, node, ("0",))
        time.sleep(0.05)
        supervisor.request_stop()
        assert result.result(timeout=2) == 1

    assert supervisor.cleanup_complete


@dataclass(slots=True)
class _FakeProcess:
    pid: int
    returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.returncode = -15
        return self.returncode


def _worker_spec(processes: tuple[NodeProcessSpec, ...]) -> NodeWorkerSpec:
    return NodeWorkerSpec(
        schema_version=1,
        resolved_gpus_per_node=8,
        nodes=(NodeSpec(node_index=0, host="compute-001", ports=(18000,), processes=processes),),
    )


def _wait_for_file(path: Path) -> None:
    deadline = time.monotonic() + 3
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists()
