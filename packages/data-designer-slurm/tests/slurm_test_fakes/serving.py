# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from enum import Enum

import httpx


class FakeServingState(str, Enum):
    """Lifecycle states shared by the serving test doubles."""

    CREATED = "created"
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    STOPPED = "stopped"


class FakeVllmBackend:
    """In-memory vLLM boundary with explicit lifecycle and responses."""

    def __init__(
        self,
        endpoint: str,
        *,
        rank: int,
        responses: Iterable[httpx.Response | BaseException] = (),
    ) -> None:
        if rank < 0:
            raise ValueError("rank must not be negative")
        self.endpoint = endpoint
        self.rank = rank
        self.state = FakeServingState.CREATED
        self.failure_reason: str | None = None
        self.requests: list[httpx.Request] = []
        self.start_calls = 0
        self.cleanup_calls = 0
        self._responses = deque(responses)

    def start(self) -> None:
        """Enter the explicit startup state."""
        self.start_calls += 1
        if self.state is FakeServingState.CREATED:
            self.state = FakeServingState.STARTING

    def mark_ready(self) -> None:
        """Mark startup complete."""
        if self.state is not FakeServingState.STARTING:
            raise RuntimeError("backend must be starting before it becomes ready")
        self.state = FakeServingState.READY

    def fail(self, reason: str) -> None:
        """Inject a backend or rank failure."""
        if not reason:
            raise ValueError("failure reason must not be empty")
        self.failure_reason = reason
        self.state = FakeServingState.FAILED

    def queue_response(self, response: httpx.Response | BaseException) -> None:
        """Queue one explicit generation response or failure."""
        self._responses.append(response)

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Handle a health or generation request without opening a socket."""
        self.requests.append(request)
        if request.url.path == "/health":
            status_code = 200 if self.state is FakeServingState.READY else 503
            return httpx.Response(status_code, request=request)
        if self.state is not FakeServingState.READY:
            return httpx.Response(503, json={"error": "backend_unavailable"}, request=request)
        if not self._responses:
            return httpx.Response(
                200,
                json={"backend": self.endpoint, "rank": self.rank},
                request=request,
            )
        response = self._responses.popleft()
        if isinstance(response, BaseException):
            raise response
        response.request = request
        return response

    def cleanup(self) -> None:
        """Stop the backend once while allowing cleanup re-entry."""
        self.cleanup_calls += 1
        self.state = FakeServingState.STOPPED


class FakeLogicalEndpoint:
    """In-memory logical endpoint over one or more fake vLLM backends."""

    def __init__(self, endpoint: str, backends: Iterable[FakeVllmBackend]) -> None:
        self.endpoint = endpoint
        self.backends = tuple(backends)
        if not self.backends:
            raise ValueError("logical endpoints require at least one backend")
        self.state = FakeServingState.CREATED
        self.published_endpoint: str | None = None
        self.requests: list[httpx.Request] = []
        self.publish_calls = 0
        self.cleanup_calls = 0
        self.failure_reason: str | None = None
        self._next_backend = 0
        self._publication_failure: Exception | None = None

    def start(self) -> None:
        """Start every backend and the readiness aggregator."""
        if self.state is not FakeServingState.CREATED:
            return
        for backend in self.backends:
            backend.start()
        self.state = FakeServingState.STARTING

    def refresh_readiness(self) -> FakeServingState:
        """Aggregate backend readiness with coordinated failure semantics."""
        if self.state is FakeServingState.STOPPED:
            return self.state
        if self._publication_failure is not None and self.publish_calls:
            self.state = FakeServingState.FAILED
            self.failure_reason = "endpoint_publication_failed"
            return self.state
        backend_states = {backend.state for backend in self.backends}
        if FakeServingState.FAILED in backend_states:
            self.state = FakeServingState.FAILED
            self.failure_reason = "backend_failed"
        elif backend_states == {FakeServingState.READY}:
            self.state = FakeServingState.READY
        else:
            self.state = FakeServingState.STARTING
        return self.state

    def publish(self) -> str:
        """Publish the logical endpoint after every backend is ready."""
        if self.refresh_readiness() is not FakeServingState.READY:
            raise RuntimeError("logical endpoint is not ready")
        self.publish_calls += 1
        if self._publication_failure is not None:
            self.state = FakeServingState.FAILED
            self.failure_reason = "endpoint_publication_failed"
            raise self._publication_failure
        self.published_endpoint = self.endpoint
        return self.endpoint

    def script_publication_failure(self, error: Exception) -> None:
        """Inject one explicit endpoint-publication failure."""
        self._publication_failure = error

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Handle an aggregated health request or forward in round-robin order."""
        self.requests.append(request)
        state = self.refresh_readiness()
        if request.url.path == "/health":
            status_code = 200 if state is FakeServingState.READY and self.published_endpoint else 503
            return httpx.Response(status_code, request=request)
        if state is FakeServingState.FAILED:
            return httpx.Response(503, json={"error": self.failure_reason}, request=request)
        if state is not FakeServingState.READY or self.published_endpoint is None:
            return httpx.Response(503, json={"error": "endpoint_unavailable"}, request=request)
        response: httpx.Response | None = None
        for _ in self.backends:
            backend = self.backends[self._next_backend]
            self._next_backend = (self._next_backend + 1) % len(self.backends)
            response = backend.handle(request)
            if response.status_code != 429:
                return response
        assert response is not None
        return response

    def cleanup(self) -> None:
        """Stop every backend and unpublish the endpoint idempotently."""
        self.cleanup_calls += 1
        if self.state is FakeServingState.STOPPED:
            return
        for backend in self.backends:
            backend.cleanup()
        self.published_endpoint = None
        self.state = FakeServingState.STOPPED
