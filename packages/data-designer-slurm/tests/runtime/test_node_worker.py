# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from data_designer.slurm.runtime import node_worker as runtime_node_worker
from data_designer.slurm.runtime.bundle import stage_runtime_bundle
from data_designer.slurm.runtime.node_spec import (
    NodeProcessSpec,
    NodeReadinessSpec,
    NodeSpec,
    NodeWorkerSpec,
    decode_node_worker_spec,
    encode_node_worker_spec,
)
from data_designer.slurm.runtime.node_worker import NodeProcessSupervisor


@pytest.fixture
def shared_gpu_spec() -> NodeWorkerSpec:
    processes = tuple(
        NodeProcessSpec(
            f"lane-{index}",
            ("serve",),
            (0,),
            0,
            NodeReadinessSpec(port=18000 + index, path="/health", deadline_seconds=1),
        )
        for index in range(2)
    )
    return NodeWorkerSpec(1, 1, None, (NodeSpec(0, "compute-001", (18000, 18001), processes),), allow_gpu_sharing=True)


@pytest.mark.parametrize(
    "mutation",
    (
        None,
        "policy",
        "boolean",
        "missing_probe",
        "remote_probe",
        "unowned_port",
        "bad_path",
        "bad_deadline",
        "bad_port_type",
        "outside",
        "unknown",
    ),
)
def test_shared_gpu_transport_requires_explicit_policy_and_valid_probes(
    shared_gpu_spec: NodeWorkerSpec,
    mutation: str | None,
) -> None:
    payload = json.loads(base64.urlsafe_b64decode(encode_node_worker_spec(shared_gpu_spec)))
    process = payload["nodes"][0]["processes"][0]
    if mutation == "policy":
        del payload["allow_gpu_sharing"]
    elif mutation == "boolean":
        payload["allow_gpu_sharing"] = "true"
    elif mutation == "missing_probe":
        del process["readiness"]
    elif mutation == "remote_probe":
        process["readiness"]["host"] = "other-host"
    elif mutation == "unowned_port":
        process["readiness"]["port"] = 18002
    elif mutation == "bad_path":
        process["readiness"]["path"] = "health"
    elif mutation == "bad_deadline":
        process["readiness"]["deadline_seconds"] = 0
    elif mutation == "bad_port_type":
        process["readiness"]["port"] = "18000"
    elif mutation == "outside":
        process["gpu_indices"] = [1]
    elif mutation == "unknown":
        payload["unknown"] = True
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    if mutation is None:
        assert decode_node_worker_spec(encoded) == shared_gpu_spec
    else:
        with pytest.raises(ValueError):
            decode_node_worker_spec(encoded)


@pytest.mark.parametrize("outcome", ("ready", "failed", "cancelled", "timeout"))
def test_shared_gpu_startup_waits_for_readiness_and_cleans_up(
    shared_gpu_spec: NodeWorkerSpec,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    children: list[_FakeProcess] = []
    events: list[str] = []
    clock = [0.0]
    supervisor = NodeProcessSupervisor(environment={})

    def start(*args: object, **kwargs: object) -> _FakeProcess:
        del args
        events.append("start")
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-remapped"
        child = _FakeProcess(pid=40 + len(children), returncode=23 if children else None)
        children.append(child)
        return child

    def probe(*args: object, **kwargs: object) -> bool:
        del args, kwargs
        events.append("probe")
        if outcome == "failed":
            children[0].returncode = 23
        elif outcome == "cancelled":
            supervisor.request_stop()
        elif outcome == "timeout":
            clock[0] = 2.0
        return outcome == "ready"

    monkeypatch.setattr(runtime_node_worker.subprocess, "Popen", start)
    monkeypatch.setattr(runtime_node_worker.HttpReadinessProber, "is_ready", probe)
    monkeypatch.setattr(runtime_node_worker.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        runtime_node_worker, "_signal_process_group", lambda child, selected: setattr(child, "returncode", -15)
    )
    if outcome == "timeout":
        with pytest.raises(RuntimeError, match="readiness timed out"):
            supervisor.run(shared_gpu_spec.nodes[0], ("GPU-remapped",))
    else:
        assert supervisor.run(shared_gpu_spec.nodes[0], ("GPU-remapped",)) == (1 if outcome == "cancelled" else 23)
    assert events == (["start", "probe", "start"] if outcome == "ready" else ["start", "probe"])
    assert supervisor.cleanup_complete
    assert all(child.poll() is not None for child in children)


@pytest.mark.parametrize("visible", ("0", "7", "GPU-remapped"))
def test_shared_gpu_real_children_wait_for_health_and_receive_the_same_device(
    shared_gpu_spec: NodeWorkerSpec,
    tmp_path: Path,
    unused_tcp_port: int,
    visible: str,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    program = f"""
import os, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

Path({str(first)!r}).write_text(os.environ['CUDA_VISIBLE_DEVICES'])
time.sleep(0.2)
HTTPServer(('127.0.0.1', {unused_tcp_port}), Handler).serve_forever()
"""
    follower = (
        "import os; from pathlib import Path; from urllib.request import urlopen; "
        f"urlopen('http://127.0.0.1:{unused_tcp_port}/health',timeout=1).close(); "
        f"Path({str(second)!r}).write_text(os.environ['CUDA_VISIBLE_DEVICES']); raise SystemExit(23)"
    )
    head, tail = shared_gpu_spec.nodes[0].processes
    assert head.readiness is not None
    head = replace(
        head,
        command=(sys.executable, "-c", program),
        readiness=replace(head.readiness, port=unused_tcp_port, deadline_seconds=3),
    )
    tail = replace(tail, command=(sys.executable, "-c", follower))
    node = replace(shared_gpu_spec.nodes[0], ports=tuple(sorted((unused_tcp_port, 18001))), processes=(head, tail))
    spec = decode_node_worker_spec(encode_node_worker_spec(replace(shared_gpu_spec, nodes=(node,))))
    supervisor = NodeProcessSupervisor(environment={})
    assert supervisor.run(spec.nodes[0], (visible,)) == 23
    assert supervisor.cleanup_complete
    assert first.read_text() == second.read_text() == visible


def test_bundled_node_worker_imports_without_client_packages(tmp_path: Path) -> None:
    workspace, extracted = tmp_path / "workspace", tmp_path / "extracted"
    workspace.mkdir(mode=0o700)
    bundle = stage_runtime_bundle(workspace)
    with tarfile.open(bundle.path) as archive:
        archive.extractall(extracted, filter="data")
    program = """
import importlib.abc
import sys

class NoClientPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in {'data_designer.config', 'data_designer.engine', 'data_designer.interface'}:
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, NoClientPackages())
from data_designer.slurm.runtime import node_worker
assert 'data_designer.slurm.planning' not in sys.modules
print(node_worker.__file__, end='')
"""
    result = subprocess.run(
        (sys.executable, "-c", program),
        env={**os.environ, "PYTHONPATH": str(extracted)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(extracted / "data_designer/slurm/runtime/node_worker.py")


def test_node_worker_spec_round_trips_and_rejects_untrusted_payload() -> None:
    spec = _worker_spec((NodeProcessSpec("lane-0", ("true",), (0,), 0),))
    assert decode_node_worker_spec(encode_node_worker_spec(spec)) == spec
    with pytest.raises(ValueError, match="invalid"):
        decode_node_worker_spec("not-base64")


def test_node_preflight_rejects_missing_required_model_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_path = tmp_path / "missing-model"
    spec = _worker_spec(
        (NodeProcessSpec("lane-0", ("true",), (0,), 0),),
        required_model_path=model_path.as_posix(),
    )
    _set_node_environment(monkeypatch)

    assert runtime_node_worker.main(("preflight", "--spec", encode_node_worker_spec(spec))) == 70
    assert capsys.readouterr().err == "node worker failed at a validated runtime boundary\n"


def test_node_preflight_rejects_unreadable_required_model_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    spec = _worker_spec(
        (NodeProcessSpec("lane-0", ("true",), (0,), 0),),
        required_model_path=model_path.as_posix(),
    )
    _set_node_environment(monkeypatch)
    monkeypatch.setattr(runtime_node_worker.os, "access", lambda path, mode: False)

    assert runtime_node_worker.main(("preflight", "--spec", encode_node_worker_spec(spec))) == 70
    assert capsys.readouterr().err == "node worker failed at a validated runtime boundary\n"


def test_partial_startup_failure_cleans_already_started_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess(pid=41)
    calls = 0
    signals: list[int] = []
    cache_paths: list[Path] = []

    def start(*arguments: object, **keywords: object) -> _FakeProcess:
        nonlocal calls
        del arguments
        environment = keywords["env"]
        assert isinstance(environment, dict)
        cache_paths.append(Path(environment["TIKTOKEN_RS_CACHE_DIR"]))
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
    assert len(cache_paths) == 2
    assert all(not path.exists() for path in cache_paths)


def test_node_worker_assigns_distinct_visible_gpus_to_two_tp4_replicas(monkeypatch: pytest.MonkeyPatch) -> None:
    visible: list[str] = []
    cache_paths: list[Path] = []

    def start(*arguments: object, **keywords: object) -> _FakeProcess:
        del arguments
        environment = keywords["env"]
        assert isinstance(environment, dict)
        visible.append(environment["CUDA_VISIBLE_DEVICES"])
        cache_path = Path(environment["TIKTOKEN_RS_CACHE_DIR"])
        assert cache_path.is_dir()
        cache_paths.append(cache_path)
        return _FakeProcess(pid=40 + len(visible), returncode=1 if len(visible) == 2 else None)

    def terminate(child: _FakeProcess, selected: int) -> None:
        del selected
        child.returncode = -15

    monkeypatch.setattr(runtime_node_worker.subprocess, "Popen", start)
    monkeypatch.setattr(runtime_node_worker, "_signal_process_group", terminate)
    node = NodeSpec(
        node_index=0,
        host="compute-001",
        ports=(),
        processes=(
            NodeProcessSpec("replica-0", ("first",), (0, 1, 2, 3), 0),
            NodeProcessSpec("replica-1", ("second",), (4, 5, 6, 7), 0),
        ),
    )

    assert NodeProcessSupervisor(environment={}).run(node, tuple(str(index) for index in range(8))) == 1
    assert visible == ["0,1,2,3", "4,5,6,7"]
    assert len(set(cache_paths)) == 2
    assert all(not path.exists() for path in cache_paths)


def test_node_worker_preserves_configured_tokenizer_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache_path = tmp_path / "prewarmed-cache"
    cache_path.mkdir()
    observed: list[str] = []

    def start(*arguments: object, **keywords: object) -> _FakeProcess:
        del arguments
        environment = keywords["env"]
        assert isinstance(environment, dict)
        observed.append(environment["TIKTOKEN_RS_CACHE_DIR"])
        return _FakeProcess(pid=41, returncode=1)

    monkeypatch.setattr(runtime_node_worker.subprocess, "Popen", start)
    node = NodeSpec(
        node_index=0,
        host="compute-001",
        ports=(),
        processes=(NodeProcessSpec("replica-0", ("serve",), (0,), 0),),
    )

    assert NodeProcessSupervisor(environment={"TIKTOKEN_RS_CACHE_DIR": cache_path.as_posix()}).run(node, ("0",)) == 1
    assert observed == [cache_path.as_posix()]
    assert cache_path.is_dir()


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


def _worker_spec(
    processes: tuple[NodeProcessSpec, ...],
    *,
    required_model_path: str | None = None,
) -> NodeWorkerSpec:
    return NodeWorkerSpec(
        schema_version=1,
        resolved_gpus_per_node=8,
        required_model_path=required_model_path,
        nodes=(NodeSpec(node_index=0, host="compute-001", ports=(18000,), processes=processes),),
    )


def _set_node_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setenv("SLURMD_NODENAME", "compute-001")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")


def _wait_for_file(path: Path) -> None:
    deadline = time.monotonic() + 3
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists()
