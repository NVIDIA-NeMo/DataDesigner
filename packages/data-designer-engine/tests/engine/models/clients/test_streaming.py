# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from data_designer.config import (
    ChatCompletionInferenceParams,
    CustomColumnConfig,
    ModelConfig,
    ModelProvider,
    custom_column_generator,
)
from data_designer.engine.column_generators.generators.custom import CustomColumnGenerator
from data_designer.engine.column_generators.utils.errors import CustomColumnGenerationError
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.clients.adapters.anthropic import AnthropicClient
from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient
from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind
from data_designer.engine.models.clients.model_request_executor import ModelRequestExecutor
from data_designer.engine.models.clients.retry import RetryConfig
from data_designer.engine.models.clients.types import ChatCompletionRequest
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.models.request_admission.controller import AdaptiveRequestAdmissionController
from data_designer.engine.models.utils import ChatMessage
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.engine.testing import InMemoryAdmissionEventSink


class FragmentedStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """Expose real HTTPX incremental decoding and observable cleanup to adapter tests."""

    def __init__(self, body: bytes, error: Exception | None = None, gate: asyncio.Event | None = None) -> None:
        """Store body bytes, optional terminal error, and optional async read gate."""
        self.body: bytes = body
        self.error: Exception | None = error
        self.gate: asyncio.Event | None = gate
        self.waiting: asyncio.Event = asyncio.Event()
        self.closed: bool = False

    def __iter__(self) -> Iterator[bytes]:
        """Yield small byte fragments, then raise the configured transport failure."""
        for start in range(0, len(self.body), 7):
            yield self.body[start : start + 7]
        if self.error:
            raise self.error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield fragments cooperatively and optionally wait for cancellation at the end."""
        for part in self:
            await asyncio.sleep(0)
            yield part
        if self.gate is not None:
            self.waiting.set()
            await self.gate.wait()

    def close(self) -> None:
        """Record that HTTPX released the synchronous response."""
        self.closed = True

    async def aclose(self) -> None:
        """Record that HTTPX released the asynchronous response."""
        self.closed = True


def encode_events(events: list[dict[str, Any] | str]) -> bytes:
    """Encode JSON events or literal strings as UTF-8 SSE, including CRLF and comments."""
    frames = ["\ufeff: keepalive\r\n\r\n"]
    for event in events:
        data = json.dumps(event, ensure_ascii=False) if isinstance(event, dict) else event
        frames.append(f"data: {data}\r\n\r\n")
    return "".join(frames).encode()


def make_chunk(delta: dict[str, Any], finish: str | None = None, index: int = 0) -> dict[str, Any]:
    """Return an indexed OpenAI chunk carrying delta fields and an optional finish reason."""
    return {"id": "chat-test", "choices": [{"index": index, "delta": delta, "finish_reason": finish}]}


def make_client(
    transport: httpx.MockTransport, is_async: bool, anthropic: bool = False
) -> OpenAICompatibleClient | AnthropicClient:
    """Return a native adapter using the supplied HTTP transport, mode, and API dialect."""
    cls = AnthropicClient if anthropic else OpenAICompatibleClient
    kwargs = (
        {"async_client": httpx.AsyncClient(transport=transport)}
        if is_async
        else {"sync_client": httpx.Client(transport=transport)}
    )
    return cls(
        provider_name="test",
        endpoint="https://example.test/v1",
        api_key="test-key",
        concurrency_mode=ClientConcurrencyMode.ASYNC if is_async else ClientConcurrencyMode.SYNC,
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_stream_assembles_interleaved_choices_tools_and_usage(is_async: bool) -> None:
    """Both execution modes preserve fragmented text/tools, choice order, and final usage."""
    events = [
        make_chunk({"role": "assistant", "content": "second"}, "length", index=1),
        make_chunk({"role": "assistant", "reasoning_content": "思"}),
        make_chunk({"reasoning_content": "考", "content": "你"}),
        make_chunk(
            {
                "content": "好",
                "tool_calls": [
                    {"index": 1, "id": "call_b", "type": "function", "function": {"name": "second", "arguments": "{"}},
                    {
                        "index": 0,
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "first", "arguments": '{"x":'},
                    },
                ],
            }
        ),
        make_chunk(
            {
                "tool_calls": [
                    {"index": 0, "function": {"arguments": "1}"}},
                    {"index": 1, "function": {"arguments": "}"}},
                ]
            },
            "tool_calls",
        ),
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 7,
                "total_tokens": 16,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        },
        "[DONE]",
    ]
    stream = FragmentedStream(encode_events(events))
    requests: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        """Capture the outgoing request body and return the fragmented SSE response."""
        requests.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.extensions["timeout"]["read"] == 19
        return httpx.Response(200, headers={"content-type": "text/event-stream; charset=utf-8"}, stream=stream)

    client = make_client(httpx.MockTransport(handle), is_async)
    request = ChatCompletionRequest("model", [{"role": "user", "content": "hello"}], stream=True, n=2, timeout=19)
    try:
        result = await client.acompletion(request) if is_async else client.completion(request)
        assert [c.message.content for c in result.choices] == ["你好", "second"]
        assert [c.finish_reason for c in result.choices] == ["tool_calls", "length"]
        assert result.message.reasoning_content == "思考"
        assert [(t.id, t.name, t.arguments_json) for t in result.message.tool_calls] == [
            ("call_a", "first", '{"x":1}'),
            ("call_b", "second", "{}"),
        ]
        assert result.usage.total_tokens == 16
        assert result.usage.reasoning_tokens == 2
        assert requests[0]["stream"] is True
        assert requests[0]["stream_options"] == {"include_usage": True}
        assert stream.closed
    finally:
        await client.aclose() if is_async else client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_anthropic_stream_preserves_blocks_signatures_and_cumulative_usage(is_async: bool) -> None:
    """Messages streaming preserves ordered content/tool blocks without summing cumulative usage."""
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        },
        {"type": "ping"},
        {"type": "future_event"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Think"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "你好"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "citations_delta", "citation": {"url": "url"}}},
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "tool_use", "id": "call", "name": "add", "input": {}},
        },
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"a":'}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "2}"}},
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": None}, "usage": {"output_tokens": 4}},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 8}},
        {"type": "message_stop"},
    ]
    stream = FragmentedStream(encode_events(events))
    client = make_client(
        httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
        ),
        is_async,
        anthropic=True,
    )
    try:
        request = ChatCompletionRequest("model", [{"role": "user", "content": "hello"}], stream=True)
        result = await client.acompletion(request) if is_async else client.completion(request)
        assert result.message.content == "你好"
        assert result.message.reasoning_content == "Think"
        assert json.loads(result.message.tool_calls[0].arguments_json) == {"a": 2}
        assert result.choices[0].finish_reason == "tool_use"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 8
        assert result.raw["content"][0]["signature"] == "sig"
        assert result.raw["content"][1]["citations"] == [{"url": "url"}]
        assert stream.closed
    finally:
        await client.aclose() if is_async else client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    ("events", "error", "kind"),
    [
        ([make_chunk({"content": "partial"})], None, ProviderErrorKind.API_CONNECTION),
        ([make_chunk({"content": "partial"}), "[DONE]"], None, ProviderErrorKind.API_CONNECTION),
        (["[DONE]"], None, ProviderErrorKind.API_CONNECTION),
        (["{malformed"], None, ProviderErrorKind.API_ERROR),
        (["[]"], None, ProviderErrorKind.API_ERROR),
        ([{"error": {"code": 429, "message": "rate limited"}}], None, ProviderErrorKind.RATE_LIMIT),
        ([make_chunk({"content": "partial"})], httpx.ReadError("disconnected"), ProviderErrorKind.API_CONNECTION),
        ([make_chunk({"content": "partial"})], httpx.ReadTimeout("idle"), ProviderErrorKind.TIMEOUT),
    ],
)
async def test_stream_failure_never_returns_partial_output(
    is_async: bool, events: list[dict[str, Any] | str], error: Exception | None, kind: ProviderErrorKind
) -> None:
    """Given malformed/error/truncated events, both modes raise the expected kind and close HTTP."""
    stream = FragmentedStream(encode_events(events), error=error)
    client = make_client(
        httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
        ),
        is_async,
    )
    try:
        request = ChatCompletionRequest("model", [], stream=True)
        with pytest.raises(ProviderError) as caught:
            await client.acompletion(request) if is_async else client.completion(request)
        assert caught.value.kind == kind
        assert caught.value.provider_name == "test"
        assert caught.value.model_name == "model"
        assert stream.closed
    finally:
        await client.aclose() if is_async else client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("anthropic", [False, True], ids=["openai", "anthropic"])
@pytest.mark.parametrize("streaming", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize("status", [200, 401, 429, 503])
async def test_http_errors_and_invalid_response_bodies(
    is_async: bool, anthropic: bool, streaming: bool, status: int
) -> None:
    """Both response paths preserve error details and release the HTTP response on failure.

    Args:
        is_async: Selects the synchronous or asynchronous HTTP client.
        anthropic: Selects the Messages protocol instead of OpenAI chat completions.
        streaming: Requests SSE instead of a buffered JSON response.
        status: HTTP error status, or 200 with a body invalid for the requested format.
    """
    stream = FragmentedStream(b"not JSON" if status == 200 else b'{"error":{"message":"failed"}}')
    client = make_client(
        httpx.MockTransport(
            lambda _: httpx.Response(
                status, headers={"content-type": "application/json", "retry-after": "3"}, stream=stream
            )
        ),
        is_async,
        anthropic=anthropic,
    )
    try:
        with pytest.raises(ProviderError) as caught:
            request = ChatCompletionRequest("model", [{"role": "user", "content": "hello"}], stream=streaming)
            await client.acompletion(request) if is_async else client.completion(request)
        assert caught.value.status_code == (None if status == 200 and streaming else status)
        assert (
            caught.value.kind
            == {
                200: ProviderErrorKind.API_ERROR,
                401: ProviderErrorKind.AUTHENTICATION,
                429: ProviderErrorKind.RATE_LIMIT,
                503: ProviderErrorKind.INTERNAL_SERVER,
            }[status]
        )
        if status != 200:
            assert "failed" in str(caught.value)
        if status == 429:
            assert caught.value.retry_after == 3
        assert stream.closed
    finally:
        await client.aclose() if is_async else client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_disconnected_stream_retries_whole_request_with_fresh_state(is_async: bool) -> None:
    """A disconnect discards partial text; the request executor reacquires admission for retry."""
    streams = [
        FragmentedStream(encode_events([make_chunk({"content": "discard"})])),
        FragmentedStream(encode_events([make_chunk({"content": "complete"}, "stop"), "[DONE]"])),
    ]
    pending = iter(streams)
    client = make_client(
        httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=next(pending))
        ),
        is_async,
    )
    sink = InMemoryAdmissionEventSink()
    controller = AdaptiveRequestAdmissionController(event_sink=sink)
    controller.register(provider_name="test", model_id="model", alias="test", max_parallel_requests=1)
    executor = ModelRequestExecutor(
        client,
        controller,
        "test",
        "model",
        event_sink=sink,
        retry_config=RetryConfig(max_retries=1, backoff_factor=0),
    )
    try:
        request = ChatCompletionRequest("model", [], stream=True)
        result = (
            await executor.acompletion(request) if is_async else await asyncio.to_thread(executor.completion, request)
        )
        assert result.message.content == "complete"
        assert all(stream.closed for stream in streams)
        completed = [event for event in sink.request_events if event.event_kind == "model_request_completed"]
        assert [event.diagnostics["outcome"] for event in completed] == ["api_connection", "success"]
    finally:
        await executor.aclose() if is_async else executor.close()


@pytest.mark.asyncio
async def test_cancelled_stream_closes_connection_and_releases_admission() -> None:
    """Cancelling a waiting SSE read closes it and records a released, cancelled request."""
    stream = FragmentedStream(encode_events([make_chunk({"content": "partial"})]), gate=asyncio.Event())
    client = make_client(
        httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
        ),
        True,
    )
    sink = InMemoryAdmissionEventSink()
    controller = AdaptiveRequestAdmissionController(event_sink=sink)
    controller.register(provider_name="test", model_id="model", alias="test", max_parallel_requests=1)
    executor = ModelRequestExecutor(
        client,
        controller,
        "test",
        "model",
        event_sink=sink,
    )
    task = asyncio.create_task(executor.acompletion(ChatCompletionRequest("model", [], stream=True)))
    try:
        await asyncio.wait_for(stream.waiting.wait(), 2)
        assert not stream.closed
        assert any(s.active_lease_count == 1 for s in controller.pressure.snapshots().values())
        assert not any(event.event_kind == "model_request_completed" for event in sink.request_events)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed
        completed = [event for event in sink.request_events if event.event_kind == "model_request_completed"]
        assert completed[0].diagnostics["outcome"] == "local_cancelled"
        assert all(s.active_lease_count == 0 for s in controller.pressure.snapshots().values())
    finally:
        task.cancel()
        await executor.aclose()


@pytest.mark.parametrize("options", [None, {"include_usage": False}, {"include_usage": True}])
def test_stream_options_respect_endpoint_overrides(options: dict[str, Any] | None) -> None:
    """Explicit stream_options can disable requesting usage or omit unsupported options entirely."""
    stream = FragmentedStream(encode_events([make_chunk({"content": "ok"}, "stop"), "[DONE]"]))
    payloads: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        """Record the supplied request and return a valid SSE response without usage."""
        payloads.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    client = make_client(httpx.MockTransport(handle), False)
    try:
        result = client.completion(
            ChatCompletionRequest("model", [], stream=True, extra_body={"stream_options": options})
        )
        assert result.message.content == "ok"
        assert result.usage is None
        if options is None:
            assert "stream_options" not in payloads[0]
        else:
            assert payloads[0]["stream_options"] == options
    finally:
        client.close()


@pytest.mark.parametrize(
    ("configured", "config_extra", "provider_extra", "kwargs", "expected"),
    [
        (True, None, None, {}, True),
        (True, None, None, {"stream": False}, False),
        (False, {"stream": True}, None, {}, True),
        (False, {"stream": True}, None, {"extra_body": None}, False),
        (True, None, {"stream": False}, {"stream": True}, False),
        (False, None, {"stream": True}, {"stream": False}, True),
        (True, None, None, {"extra_body": {"stream": False}}, False),
    ],
)
def test_facade_stream_flag_matches_effective_wire_request(
    configured: bool, config_extra: dict | None, provider_extra: dict | None, kwargs: dict, expected: bool
) -> None:
    """Model, call, extra-body, and provider overrides resolve identically for bridges and HTTP."""
    provider = ModelProvider(
        name="test", endpoint="https://example.test/v1", provider_type="openai", extra_body=provider_extra
    )
    config = ModelConfig(
        alias="test",
        model="model",
        provider="test",
        inference_parameters=ChatCompletionInferenceParams(
            stream=configured,
            extra_body=config_extra,
        ),
    )

    def handle(request: httpx.Request) -> httpx.Response:
        """Assert the outgoing request's mode and return the corresponding wire response format."""
        payload = json.loads(request.content)
        assert bool(payload.get("stream")) is expected
        if expected:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=FragmentedStream(encode_events([make_chunk({"content": "ok"}, "stop"), "[DONE]"])),
            )
        return httpx.Response(
            200, json={"choices": [{"index": 0, "message": {"content": "ok"}, "finish_reason": "stop"}]}
        )

    client = make_client(httpx.MockTransport(handle), False)
    model = ModelFacade(config, ModelProviderRegistry(providers=[provider]), client=client)
    try:
        assert model.is_streaming_enabled(**kwargs) is expected
        assert model.completion([ChatMessage.as_user("hello")], **kwargs).message.content == "ok"
    finally:
        model.close()


class DelayedStream(FragmentedStream):
    """A stream whose total duration exceeds the test's synthetic bridge deadline."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield a heartbeat, await a slow response, then yield complete SSE data."""
        yield b": keepalive\n\n"
        await asyncio.sleep(0.05)
        yield self.body


@custom_column_generator(model_aliases=["test"])
def generate_streamed_custom_row(row: dict, generator_params: Any, models: dict[str, Any]) -> dict:
    """Generate result into row using model 'test'; generator_params is unused."""
    row["result"], _ = models["test"].generate("Say ok")
    return row


@pytest.mark.asyncio
@pytest.mark.parametrize("completion_races_with_timeout", [False, True])
async def test_sync_custom_column_waits_for_active_stream(
    monkeypatch: pytest.MonkeyPatch, completion_races_with_timeout: bool
) -> None:
    """A synchronous custom column returns a complete async response despite bridge polling deadlines.

    Args:
        monkeypatch: Shortens the bridge deadline and optionally injects a polling race.
        completion_races_with_timeout: Completes generation just before reporting a polling timeout.
    """
    if completion_races_with_timeout:
        future_result = concurrent.futures.Future.result

        def finish_before_timeout_check(future: concurrent.futures.Future, timeout: float | None = None) -> Any:
            """Read future using timeout, then simulate a polling timeout racing with its completed result."""
            result = future_result(future, timeout)
            if timeout is not None:
                raise concurrent.futures.TimeoutError
            return result

        monkeypatch.setattr(concurrent.futures.Future, "result", finish_before_timeout_check)

    stream = DelayedStream(encode_events([make_chunk({"content": "ok"}, "stop"), "[DONE]"]))
    client = make_client(
        httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=stream,
            )
        ),
        True,
    )
    config = ModelConfig(
        alias="test", model="model", provider="test", inference_parameters=ChatCompletionInferenceParams(stream=True)
    )
    provider = ModelProvider(name="test", endpoint="https://example.test/v1", provider_type="openai")
    model = ModelFacade(config, ModelProviderRegistry(providers=[provider]), client=client)
    resources = MagicMock(spec=ResourceProvider)
    resources.model_registry = MagicMock()
    resources.model_registry.get_model.return_value = model
    generator = CustomColumnGenerator(
        config=CustomColumnConfig(name="result", generator_function=generate_streamed_custom_row),
        resource_provider=resources,
    )
    monkeypatch.setattr(
        "data_designer.engine.column_generators.generators.custom._compute_bridge_timeout", lambda *args: 0.01
    )
    try:
        assert await asyncio.wait_for(asyncio.to_thread(generator.generate, {}), 2) == {"result": "ok"}
        assert stream.closed
    finally:
        await model.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    ("events", "kind"),
    [
        ([{"type": "message_stop"}], ProviderErrorKind.API_CONNECTION),
        ([{"type": "message_start", "message": {"content": [], "usage": {}}}], ProviderErrorKind.API_CONNECTION),
        (
            [{"type": "error", "error": {"type": "rate_limit_error", "message": "limited"}}],
            ProviderErrorKind.RATE_LIMIT,
        ),
        (
            [{"type": "error", "error": {"type": "overloaded_error", "message": "overloaded"}}],
            ProviderErrorKind.INTERNAL_SERVER,
        ),
        (
            [{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lost"}}],
            ProviderErrorKind.API_ERROR,
        ),
        (
            [
                {"type": "message_start", "message": {"content": [], "usage": {}}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "partial"}},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
                {"type": "message_stop"},
            ],
            ProviderErrorKind.API_CONNECTION,
        ),
        (
            [
                {"type": "message_start", "message": {"content": [], "usage": {}}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "input": {}}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{"}},
                {"type": "content_block_stop", "index": 0},
            ],
            ProviderErrorKind.API_ERROR,
        ),
    ],
)
async def test_anthropic_stream_rejects_errors_and_incomplete_blocks(
    is_async: bool, events: list[dict[str, Any]], kind: ProviderErrorKind
) -> None:
    """Given SSE events and expected error kind, each execution mode rejects incomplete Messages."""
    stream = FragmentedStream(encode_events(events))
    client = make_client(
        httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=stream,
            )
        ),
        is_async,
        anthropic=True,
    )
    try:
        request = ChatCompletionRequest("model", [{"role": "user", "content": "hello"}], stream=True)
        with pytest.raises(ProviderError) as caught:
            await client.acompletion(request) if is_async else client.completion(request)
        assert caught.value.kind == kind
        assert stream.closed
    finally:
        await client.aclose() if is_async else client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_sse_multiline_data_and_final_chunk_usage(is_async: bool) -> None:
    """Each mode handles multiline SSE data and usage attached to the last choice chunk."""
    stream = FragmentedStream(
        b'event: message\n: comment\ndata: {"choices":\ndata: [{"index":0,"delta":{"content":"ok"},'
        b'"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}\n\n'
        b"data: [DONE]\n\n"
    )
    client = make_client(
        httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=stream,
            )
        ),
        is_async,
    )
    try:
        request = ChatCompletionRequest("model", [], stream=True)
        result = await client.acompletion(request) if is_async else client.completion(request)
        assert result.message.content == "ok"
        assert result.usage.total_tokens == 5
        assert stream.closed
    finally:
        await client.aclose() if is_async else client.close()


@pytest.mark.asyncio
async def test_streaming_custom_column_propagates_coroutine_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed coroutine's TimeoutError must not be mistaken for an endless polling timeout.

    Args:
        monkeypatch: Substitutes an async model boundary raising TimeoutError immediately.
    """
    client = make_client(httpx.MockTransport(lambda _: httpx.Response(500)), True)
    config = ModelConfig(
        alias="test", model="model", provider="test", inference_parameters=ChatCompletionInferenceParams(stream=True)
    )
    provider = ModelProvider(name="test", endpoint="https://example.test/v1", provider_type="openai")
    model = ModelFacade(config, ModelProviderRegistry(providers=[provider]), client=client)
    resources = MagicMock(spec=ResourceProvider)
    resources.model_registry = MagicMock()
    resources.model_registry.get_model.return_value = model
    generator = CustomColumnGenerator(
        config=CustomColumnConfig(name="result", generator_function=generate_streamed_custom_row),
        resource_provider=resources,
    )

    async def fail_generation(*args: Any, **kwargs: Any) -> tuple[Any, list]:
        """Reject forwarded generation args/kwargs with a coroutine-level timeout, returning no result."""
        raise TimeoutError("coroutine timed out")

    monkeypatch.setattr(model, "agenerate", fail_generation)
    try:
        with pytest.raises(CustomColumnGenerationError, match="coroutine timed out"):
            await asyncio.wait_for(asyncio.to_thread(generator.generate, {}), 2)
    finally:
        await model.aclose()
