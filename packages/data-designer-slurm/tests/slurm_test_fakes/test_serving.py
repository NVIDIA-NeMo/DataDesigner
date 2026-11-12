# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import httpx
import pytest

from slurm_test_fakes import FakeLogicalEndpoint, FakeServingState, FakeVllmBackend


def _client(endpoint: FakeLogicalEndpoint) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(endpoint.handle), base_url=endpoint.endpoint)


def _start_and_publish(endpoint: FakeLogicalEndpoint) -> None:
    endpoint.start()
    for backend in endpoint.backends:
        backend.mark_ready()
    endpoint.publish()


def test_single_backend_startup_readiness_and_endpoint_publication() -> None:
    backend = FakeVllmBackend("http://127.0.0.1:31001", rank=0)
    endpoint = FakeLogicalEndpoint("http://127.0.0.1:31000", (backend,))

    endpoint.start()
    with _client(endpoint) as client:
        assert client.get("/health").status_code == 503

        backend.mark_ready()
        assert endpoint.publish() == endpoint.endpoint
        assert client.get("/health").status_code == 200
        assert client.post("/v1/chat/completions", json={"prompt": "fixture"}).json() == {
            "backend": backend.endpoint,
            "rank": 0,
        }


def test_multiple_backends_receive_requests_in_deterministic_order(
    fake_logical_endpoint: FakeLogicalEndpoint,
) -> None:
    _start_and_publish(fake_logical_endpoint)

    with _client(fake_logical_endpoint) as client:
        responses = [client.post("/v1/chat/completions").json() for _ in range(3)]

    assert [response["rank"] for response in responses] == [0, 1, 0]
    assert [len(backend.requests) for backend in fake_logical_endpoint.backends] == [2, 1]


def test_partial_multi_backend_startup_is_not_ready_and_cleans_up_once(
    fake_logical_endpoint: FakeLogicalEndpoint,
) -> None:
    fake_logical_endpoint.start()
    fake_logical_endpoint.backends[0].mark_ready()

    assert fake_logical_endpoint.refresh_readiness() is FakeServingState.STARTING
    with pytest.raises(RuntimeError, match="not ready"):
        fake_logical_endpoint.publish()

    fake_logical_endpoint.cleanup()
    fake_logical_endpoint.cleanup()
    assert [backend.cleanup_calls for backend in fake_logical_endpoint.backends] == [1, 1]


@pytest.mark.parametrize("retry_after", ("2", None))
def test_logical_endpoint_preserves_overload_response(retry_after: str | None) -> None:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    backend = FakeVllmBackend(
        "http://127.0.0.1:31001",
        rank=0,
        responses=(httpx.Response(429, headers=headers, json={"error": "overloaded"}),),
    )
    endpoint = FakeLogicalEndpoint("http://127.0.0.1:31000", (backend,))
    _start_and_publish(endpoint)

    with _client(endpoint) as client:
        response = client.post("/v1/chat/completions")

    assert response.status_code == 429
    assert response.headers.get("Retry-After") == retry_after
    assert response.json() == {"error": "overloaded"}


def test_logical_endpoint_retries_overload_against_another_backend() -> None:
    first = FakeVllmBackend(
        "http://127.0.0.1:31001",
        rank=0,
        responses=(httpx.Response(429, json={"error": "first overloaded"}),),
    )
    second = FakeVllmBackend("http://127.0.0.1:31002", rank=1)
    endpoint = FakeLogicalEndpoint("http://127.0.0.1:31000", (first, second))
    _start_and_publish(endpoint)

    with _client(endpoint) as client:
        response = client.post("/v1/chat/completions")

    assert response.status_code == 200
    assert response.json()["rank"] == 1
    assert [len(backend.requests) for backend in endpoint.backends] == [1, 1]


def test_logical_endpoint_returns_the_final_overload_response() -> None:
    first = FakeVllmBackend(
        "http://127.0.0.1:31001",
        rank=0,
        responses=(httpx.Response(429, headers={"Retry-After": "1"}),),
    )
    second = FakeVllmBackend(
        "http://127.0.0.1:31002",
        rank=1,
        responses=(httpx.Response(429, headers={"Retry-After": "3"}),),
    )
    endpoint = FakeLogicalEndpoint("http://127.0.0.1:31000", (first, second))
    _start_and_publish(endpoint)

    with _client(endpoint) as client:
        response = client.post("/v1/chat/completions")

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "3"


def test_backend_rank_failure_fails_the_logical_endpoint(
    fake_logical_endpoint: FakeLogicalEndpoint,
) -> None:
    _start_and_publish(fake_logical_endpoint)
    fake_logical_endpoint.backends[1].fail("rank exited")

    with _client(fake_logical_endpoint) as client:
        response = client.post("/v1/chat/completions")

    assert fake_logical_endpoint.refresh_readiness() is FakeServingState.FAILED
    assert response.status_code == 503
    assert response.json() == {"error": "backend_failed"}


def test_endpoint_publication_failure_is_explicit_and_cleanup_remains_idempotent(
    fake_logical_endpoint: FakeLogicalEndpoint,
) -> None:
    fake_logical_endpoint.start()
    for backend in fake_logical_endpoint.backends:
        backend.mark_ready()
    fake_logical_endpoint.script_publication_failure(RuntimeError("publication failed"))

    with pytest.raises(RuntimeError, match="publication failed"):
        fake_logical_endpoint.publish()
    assert fake_logical_endpoint.refresh_readiness() is FakeServingState.FAILED
    assert fake_logical_endpoint.failure_reason == "endpoint_publication_failed"

    with _client(fake_logical_endpoint) as client:
        response = client.post("/v1/chat/completions")
    assert response.json() == {"error": "endpoint_publication_failed"}

    fake_logical_endpoint.cleanup()
    fake_logical_endpoint.cleanup()
    assert [backend.cleanup_calls for backend in fake_logical_endpoint.backends] == [1, 1]


def test_backend_failure_injection_is_explicit() -> None:
    backend = FakeVllmBackend(
        "http://127.0.0.1:31001",
        rank=0,
        responses=(RuntimeError("scripted backend failure"),),
    )
    endpoint = FakeLogicalEndpoint("http://127.0.0.1:31000", (backend,))
    _start_and_publish(endpoint)

    with _client(endpoint) as client, pytest.raises(RuntimeError, match="scripted backend failure"):
        client.post("/v1/chat/completions")


@pytest.mark.parametrize("terminal_state", (FakeServingState.FAILED, FakeServingState.STOPPED))
def test_backend_rejects_restart_from_terminal_state(terminal_state: FakeServingState) -> None:
    backend = FakeVllmBackend("http://127.0.0.1:31001", rank=0)
    if terminal_state is FakeServingState.FAILED:
        backend.fail("rank exited")
    else:
        backend.cleanup()

    with pytest.raises(RuntimeError, match=f"cannot start from {terminal_state.value}"):
        backend.start()


@pytest.mark.parametrize("terminal_state", (FakeServingState.FAILED, FakeServingState.STOPPED))
def test_logical_endpoint_rejects_restart_from_terminal_state(terminal_state: FakeServingState) -> None:
    endpoint = FakeLogicalEndpoint(
        "http://127.0.0.1:31000",
        (FakeVllmBackend("http://127.0.0.1:31001", rank=0),),
    )
    if terminal_state is FakeServingState.FAILED:
        endpoint.start()
        endpoint.backends[0].fail("rank exited")
        endpoint.refresh_readiness()
    else:
        endpoint.cleanup()

    with pytest.raises(RuntimeError, match=f"cannot start from {terminal_state.value}"):
        endpoint.start()


def test_cleanup_is_idempotent_and_unpublishes_endpoint(
    fake_logical_endpoint: FakeLogicalEndpoint,
) -> None:
    _start_and_publish(fake_logical_endpoint)

    fake_logical_endpoint.cleanup()
    fake_logical_endpoint.cleanup()

    assert fake_logical_endpoint.state is FakeServingState.STOPPED
    assert fake_logical_endpoint.published_endpoint is None
    assert fake_logical_endpoint.cleanup_calls == 2
    assert [backend.cleanup_calls for backend in fake_logical_endpoint.backends] == [1, 1]
