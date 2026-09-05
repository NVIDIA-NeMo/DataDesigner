# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.utils.type_helpers import StrEnum
from data_designer.engine.models.clients.adapters.http_helpers import (
    parse_json_body,
    resolve_timeout,
    wrap_transport_error,
)
from data_designer.engine.models.clients.errors import (
    ProviderError,
    ProviderErrorKind,
    SyncClientUnavailableError,
    map_http_error_to_provider_error,
)
from data_designer.engine.models.clients.retry import RetryConfig, RetryTransport, create_retry_transport

if TYPE_CHECKING:
    import httpx

    from data_designer.engine.models.clients.streaming import ChatStream


class ClientConcurrencyMode(StrEnum):
    SYNC = "sync"
    ASYNC = "async"


_POOL_MAX_MULTIPLIER = 2
_MIN_MAX_CONNECTIONS = 32
_MIN_KEEPALIVE_CONNECTIONS = 16


class HttpModelClient(ABC):
    """Shared HTTP transport and lifecycle logic for native model adapters.

    Each instance operates in exactly one mode — ``"sync"`` or ``"async"`` —
    set at construction time.  The mode determines which httpx client and
    transport teardown path is used.  Calling the wrong-mode methods raises
    ``RuntimeError`` immediately, preventing accidental dual-mode usage that
    leads to transport leaks and cross-mode teardown complexity.
    """

    def __init__(
        self,
        *,
        provider_name: str,
        endpoint: str,
        api_key: str | None = None,
        retry_config: RetryConfig | None = None,
        max_parallel_requests: int = 32,
        timeout_s: float = 60.0,
        concurrency_mode: ClientConcurrencyMode = ClientConcurrencyMode.SYNC,
        transport: RetryTransport | None = None,
        sync_client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        if concurrency_mode == ClientConcurrencyMode.SYNC and async_client is not None:
            raise ValueError("async_client must not be provided for a sync-mode HttpModelClient")
        if concurrency_mode == ClientConcurrencyMode.ASYNC and sync_client is not None:
            raise ValueError("sync_client must not be provided for an async-mode HttpModelClient")

        self.provider_name = provider_name
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._retry_config = retry_config
        self._mode: ClientConcurrencyMode = concurrency_mode

        pool_max = max(_MIN_MAX_CONNECTIONS, _POOL_MAX_MULTIPLIER * max_parallel_requests)
        pool_keepalive = max(_MIN_KEEPALIVE_CONNECTIONS, max_parallel_requests)
        self._limits = lazy.httpx.Limits(
            max_connections=pool_max,
            max_keepalive_connections=pool_keepalive,
        )
        self._transport: RetryTransport | None = transport
        self._client: httpx.Client | None = sync_client
        self._aclient: httpx.AsyncClient | None = async_client
        self._init_lock = threading.Lock()
        self._closed = False

    @property
    def concurrency_mode(self) -> ClientConcurrencyMode:
        return self._mode

    @property
    def limits(self) -> httpx.Limits:
        """Connection pool limits derived from ``max_parallel_requests`` at construction time."""
        return self._limits

    @abstractmethod
    def _build_headers(self, extra_headers: dict[str, str]) -> dict[str, str]:
        """Build provider-specific request headers."""

    # --- lazy client initialization ---

    def _get_sync_client(self) -> httpx.Client:
        if self._mode != ClientConcurrencyMode.SYNC:
            raise SyncClientUnavailableError("Sync methods are not available on an async-mode HttpModelClient.")
        with self._init_lock:
            if self._closed:
                raise RuntimeError("Model client is closed.")
            if self._client is None:
                if self._transport is None:
                    inner = lazy.httpx.HTTPTransport(limits=self._limits)
                    self._transport = create_retry_transport(
                        self._retry_config, strip_rate_limit_codes=False, transport=inner
                    )
                self._client = lazy.httpx.Client(
                    transport=self._transport,
                    timeout=lazy.httpx.Timeout(self._timeout_s),
                )
            return self._client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._mode != ClientConcurrencyMode.ASYNC:
            raise RuntimeError("Async methods are not available on a sync-mode HttpModelClient.")
        with self._init_lock:
            if self._closed:
                raise RuntimeError("Model client is closed.")
            if self._aclient is None:
                if self._transport is None:
                    inner = lazy.httpx.AsyncHTTPTransport(limits=self._limits)
                    self._transport = create_retry_transport(
                        self._retry_config, strip_rate_limit_codes=True, transport=inner
                    )
                self._aclient = lazy.httpx.AsyncClient(
                    transport=self._transport,
                    timeout=lazy.httpx.Timeout(self._timeout_s),
                )
            return self._aclient

    # --- lifecycle ---

    def close(self) -> None:
        """Release sync-mode resources.  No-op if this is an async-mode client."""
        if self._mode != ClientConcurrencyMode.SYNC:
            return
        with self._init_lock:
            client = self._client
            transport = self._transport
            self._closed = True
            self._client = None
            self._transport = None
        if client is not None:
            client.close()
        elif transport is not None:
            transport.close()

    async def aclose(self) -> None:
        """Release async-mode resources.  No-op if this is a sync-mode client."""
        if self._mode != ClientConcurrencyMode.ASYNC:
            return
        with self._init_lock:
            async_client = self._aclient
            transport = self._transport
            self._closed = True
            self._aclient = None
            self._transport = None
        if async_client is not None:
            await async_client.aclose()
        elif transport is not None:
            await transport.aclose()

    # --- HTTP helpers ---

    def _post_sync(
        self,
        route: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str],
        model_name: str,
        timeout: float | None = None,
        *,
        stream: ChatStream | None = None,
    ) -> dict[str, Any]:
        """POST JSON or SSE and return the complete response, closing the connection.

        Args:
            route: Provider route appended to the configured endpoint.
            payload: Provider-specific JSON request body.
            extra_headers: Per-request headers merged with provider authentication.
            model_name: Model identifier for errors.
            timeout: Optional per-operation timeout in seconds, including read inactivity.
            stream: Fresh accumulator for SSE; None selects an ordinary JSON response.

        Returns:
            Decoded JSON or the assembled streaming response.

        Raises:
            ProviderError: For HTTP errors, malformed events, timeouts, or interrupted streams.
        """
        client = self._get_sync_client()
        headers = self._build_headers(extra_headers)
        if stream is not None:
            headers = {"Accept": "text/event-stream", **headers}
        url = f"{self._endpoint}{route}"
        with self._request_errors(model_name, stream):
            request_kwargs: dict[str, Any] = {
                "json": payload,
                "headers": headers,
                "timeout": resolve_timeout(self._timeout_s, timeout),
            }
            if stream is None:
                response = client.post(url, **request_kwargs)
                self._validate_response(response, model_name, stream)
                return parse_json_body(response, self.provider_name, model_name)
            with client.stream("POST", url, **request_kwargs) as response:
                if response.status_code >= 400:
                    response.read()
                self._validate_response(response, model_name, stream)
                for line in response.iter_lines():
                    if stream.feed_line(line):
                        break
                return stream.finish()

    async def _apost(
        self,
        route: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str],
        model_name: str,
        timeout: float | None = None,
        *,
        stream: ChatStream | None = None,
    ) -> dict[str, Any]:
        """Asynchronously POST JSON or SSE and return the complete provider response.

        Args:
            route: Provider route appended to the configured endpoint.
            payload: Provider-specific JSON request body.
            extra_headers: Per-request headers merged with provider authentication.
            model_name: Model identifier for errors.
            timeout: Optional per-operation timeout in seconds, including read inactivity.
            stream: Fresh accumulator for SSE; None selects an ordinary JSON response.

        Returns:
            Decoded JSON or the assembled streaming response, with the connection closed.

        Raises:
            ProviderError: For HTTP, protocol, timeout, or connection failures.
            asyncio.CancelledError: Cancellation propagates after closing the response.
        """
        client = self._get_async_client()
        headers = self._build_headers(extra_headers)
        if stream is not None:
            headers = {"Accept": "text/event-stream", **headers}
        url = f"{self._endpoint}{route}"
        with self._request_errors(model_name, stream):
            request_kwargs: dict[str, Any] = {
                "json": payload,
                "headers": headers,
                "timeout": resolve_timeout(self._timeout_s, timeout),
            }
            if stream is None:
                response = await client.post(url, **request_kwargs)
                self._validate_response(response, model_name, stream)
                return parse_json_body(response, self.provider_name, model_name)
            async with client.stream("POST", url, **request_kwargs) as response:
                if response.status_code >= 400:
                    await response.aread()
                self._validate_response(response, model_name, stream)
                async for line in response.aiter_lines():
                    if stream.feed_line(line):
                        break
                return stream.finish()

    def _validate_response(
        self,
        response: httpx.Response,
        model_name: str,
        stream: ChatStream | None,
    ) -> None:
        """Check HTTP status and set UTF-8 decoding for a successful SSE response.

        Args:
            response: HTTP response, with its body already read when the status is an error.
            model_name: Model identifier attached to HTTP errors.
            stream: SSE accumulator, or None to skip SSE checks for a JSON response.

        Raises:
            ProviderError: For HTTP failures or an unexpected SSE content type.
        """
        if response.status_code >= 400:
            raise map_http_error_to_provider_error(
                response=response, provider_name=self.provider_name, model_name=model_name
            )
        if stream is not None:
            if response.headers.get("content-type", "").split(";")[0].strip().lower() != "text/event-stream":
                stream.fail("Expected text/event-stream for a streaming chat request")
            response.encoding = "utf-8"

    @contextmanager
    def _request_errors(self, model_name: str, stream: ChatStream | None) -> Iterator[None]:
        """Normalize request failures for both execution modes, leaving cancellation untouched.

        Args:
            model_name: Model identifier attached to normalized transport errors.
            stream: SSE accumulator; its transport failures must retry the complete request.

        Yields:
            Control to HTTP I/O and parsing; failures leave the context as ProviderError.
        """
        try:
            yield
        except ProviderError:
            raise
        except Exception as exc:
            if (
                stream is not None
                and isinstance(exc, lazy.httpx.TransportError)
                and not isinstance(exc, lazy.httpx.TimeoutException)
            ):
                stream.fail("Chat stream connection interrupted", kind=ProviderErrorKind.API_CONNECTION, cause=exc)
            raise wrap_transport_error(exc, self.provider_name, model_name) from exc
