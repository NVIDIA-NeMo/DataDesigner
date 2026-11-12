# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import FakeStateStore, RuntimeCase

from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.runtime import entrypoint
from data_designer.slurm.runtime.errors import SlurmRuntimeError
from data_designer.slurm.runtime.health import monitor_runtime_health, probe_runtime_health
from data_designer.slurm.state import (
    AttemptLifecycleState,
    AttemptReadiness,
    DeploymentReadiness,
    EndpointPublicationState,
    ReadinessState,
)


@pytest.fixture
def healthy_readiness(runtime_case: RuntimeCase) -> AttemptReadiness:
    attempt = runtime_case.context.attempt
    now = datetime.now(UTC)
    return AttemptReadiness(
        schema_version=1,
        run_id=attempt.run_id,
        shard_id=attempt.shard_id,
        attempt_id=attempt.attempt_id,
        revision=1,
        updated_at=now,
        state=ReadinessState.READY,
        deployments=(
            DeploymentReadiness(
                deployment_id="deployment-00000",
                model_alias="generator",
                state=ReadinessState.READY,
                expected_backends=2,
                ready_backends=2,
                endpoint_publication=EndpointPublicationState.PUBLISHED,
            ),
        ),
    )


@contextmanager
def _health_endpoint(responses: list[tuple[int, bytes]]) -> Iterator[int]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            assert self.path == "/health"
            status, body = responses[0] if len(responses) == 1 else responses.pop(0)
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_monitor_detects_health_loss_without_process_exit_and_preserves_failure_evidence(
    runtime_case: RuntimeCase,
    healthy_readiness: AttemptReadiness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = FakeStateStore(runtime_case.context.attempt)
    state.write_readiness(healthy_readiness)
    responses = [
        (200, b'{"backends":2,"backends_ready":2}'),
        (503, b'{"backends":2,"backends_ready":1}'),
    ]
    with _health_endpoint(responses) as port:
        with pytest.raises(SlurmRuntimeError, match="lost health"):
            monitor_runtime_health(state, healthy_readiness, {"deployment-00000": port}, poll_interval_seconds=0.01)
        assert state.readiness[-2].state is ReadinessState.READY
        assert state.readiness[-1].state is ReadinessState.FAILED
        assert state.readiness[-1].deployments[0].ready_backends == 1
        # The endpoint is still alive and responsive after the health transition.
        assert probe_runtime_health(healthy_readiness, {"deployment-00000": port}).state is ReadinessState.FAILED

    monkeypatch.setattr(entrypoint, "load_allocation_context", lambda *args, **kwargs: (runtime_case.context, state))
    assert (
        entrypoint.main(
            [
                "fail",
                "--plan",
                str(runtime_case.workspace / "plan.json"),
                "--attempt-dir",
                str(runtime_case.context.attempt_directory),
            ]
        )
        == 0
    )
    assert state.attempt.state is AttemptLifecycleState.FAILED
    assert state.readiness[-1].state is ReadinessState.STOPPED
    assert state.readiness[-1].deployments[0].last_probe.reason_code == "runtime_health_lost"


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (500, b"internal"),
        (200, b"not json"),
        (200, b"[]"),
        (200, b'{"backends":2,"backends_ready":true}'),
        (200, b'{"backends":2,"backends_ready":3}'),
        (200, b'{"backends":3,"backends_ready":2}'),
        (503, b'{"backends":2,"backends_ready":2}'),
        (200, b" " * 4097),
    ],
)
def test_invalid_health_response_fails_closed(healthy_readiness: AttemptReadiness, status: int, body: bytes) -> None:
    with _health_endpoint([(status, body)]) as port:
        result = probe_runtime_health(healthy_readiness, {"deployment-00000": port})
    assert result.state is ReadinessState.FAILED
    assert result.deployments[0].ready_backends == 0
    assert body.decode() not in result.model_dump_json()


def test_health_detects_endpoint_connection_loss(healthy_readiness: AttemptReadiness) -> None:
    with _health_endpoint([(200, b'{"backends":2,"backends_ready":2}')]) as port:
        assert probe_runtime_health(healthy_readiness, {"deployment-00000": port}).state is ReadinessState.READY
    result = probe_runtime_health(healthy_readiness, {"deployment-00000": port})
    assert result.state is ReadinessState.FAILED
    assert result.deployments[0].ready_backends == 0


@pytest.mark.parametrize("already_stopped", (False, True))
def test_success_rejects_health_failure_recorded_during_cleanup(
    runtime_case: RuntimeCase,
    healthy_readiness: AttemptReadiness,
    monkeypatch: pytest.MonkeyPatch,
    already_stopped: bool,
) -> None:
    state = FakeStateStore(
        runtime_case.context.attempt.model_copy(
            update={
                "state": AttemptLifecycleState.RUNNING,
                "candidate_output": ArtifactReference(
                    path=(runtime_case.context.attempt_directory / "output-manifest.json").as_posix(),
                    sha256="a" * 64,
                ),
            }
        )
    )
    state.write_readiness(healthy_readiness)
    with _health_endpoint([(503, b'{"backends":2,"backends_ready":1}')]) as port:
        state.write_readiness(probe_runtime_health(healthy_readiness, {"deployment-00000": port}))
    monkeypatch.setattr(entrypoint, "load_allocation_context", lambda *args, **kwargs: (runtime_case.context, state))
    if already_stopped:
        entrypoint._write_stopped_readiness(runtime_case.context, state)
    failure = state.readiness[-1]
    arguments = [
        "--plan",
        str(runtime_case.workspace / "plan.json"),
        "--attempt-dir",
        str(runtime_case.context.attempt_directory),
    ]
    assert entrypoint.main(["succeed", *arguments]) == 70
    assert not state.winners
    assert state.attempt.state is AttemptLifecycleState.RUNNING
    assert state.readiness[-1] is failure
    assert entrypoint.main(["fail", *arguments]) == 0
    assert state.attempt.state is AttemptLifecycleState.FAILED
    assert state.readiness[-1].deployments[0].last_probe.reason_code == "runtime_health_lost"


@pytest.mark.parametrize("invalid_interval", (False, True))
def test_monitor_rejects_invalid_settings(
    runtime_case: RuntimeCase, healthy_readiness: AttemptReadiness, invalid_interval: bool
) -> None:
    state = FakeStateStore(runtime_case.context.attempt)
    endpoints = {"deployment-00000": 1} if invalid_interval else {}
    with pytest.raises(ValueError, match="interval|endpoints"):
        monitor_runtime_health(state, healthy_readiness, endpoints, poll_interval_seconds=0 if invalid_interval else 1)
