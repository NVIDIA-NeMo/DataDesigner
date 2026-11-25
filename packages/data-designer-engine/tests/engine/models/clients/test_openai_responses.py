# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest
from pytest_httpx import HTTPXMock

from data_designer.config.mcp import MCPProvider, ToolConfig
from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig, ModelProvider
from data_designer.engine.mcp.facade import MCPFacade
from data_designer.engine.mcp.registry import MCPToolDefinition, MCPToolResult
from data_designer.engine.model_provider import MCPProviderRegistry, ModelProviderRegistry
from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
from data_designer.engine.models.clients.adapters.openai_responses import OpenAIResponsesClient
from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind
from data_designer.engine.models.clients.factory import create_model_client
from data_designer.engine.models.clients.types import ChatCompletionRequest, ProviderReplayData
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.models.factory import create_model_registry
from data_designer.engine.models.utils import ChatMessage
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.engine.testing import StubMCPRegistry


@pytest.fixture
def responses_registry() -> ModelProviderRegistry:
    return ModelProviderRegistry(
        providers=[ModelProvider(name="responses", provider_type="openai", endpoint="https://api.example.com/v1")]
    )


@pytest.fixture
def responses_config() -> ModelConfig:
    return ModelConfig(
        alias="responses",
        api="responses",
        provider="responses",
        model="gpt-test",
        inference_parameters=ChatCompletionInferenceParams(max_tokens=1024, extra_body={"reasoning_effort": "medium"}),
    )


@pytest.fixture
def responses_client() -> Iterator[OpenAIResponsesClient]:
    client = OpenAIResponsesClient(provider_name="responses", endpoint="https://api.example.com/v1")
    yield client
    client.close()


def make_response(output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "output_tokens_details": {"reasoning_tokens": 3},
        },
    }


def make_text(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_facade_replays_multiple_mcp_rounds(
    httpx_mock: HTTPXMock, responses_registry: ModelProviderRegistry, responses_config: ModelConfig, asynchronous: bool
) -> None:
    rounds = []
    for turn in range(2):
        rounds.append(
            [
                {"type": "reasoning", "id": f"rs_{turn}", "summary": [], "encrypted_content": f"encrypted-{turn}"},
                make_text(f"Checking {turn}"),
                *[
                    {
                        "type": "function_call",
                        "id": f"fc_{turn}_{i}",
                        "call_id": f"call_{turn}_{i}",
                        "name": "lookup",
                        "arguments": "{}",
                    }
                    for i in range(2)
                ],
            ]
        )
    for output in [*rounds, [make_text("done")]]:
        httpx_mock.add_response(url="https://api.example.com/v1/responses", json=make_response(output))
    mode = ClientConcurrencyMode.ASYNC if asynchronous else ClientConcurrencyMode.SYNC
    client = create_model_client(
        responses_config, PlaintextResolver(), responses_registry, client_concurrency_mode=mode
    )
    assert isinstance(client, OpenAIResponsesClient)
    mcp = MCPFacade(
        ToolConfig(tool_alias="tools", providers=["tools"]),
        PlaintextResolver(),
        MCPProviderRegistry(providers=[MCPProvider(name="tools", endpoint="https://tools.example.com")]),
    )
    facade = ModelFacade(responses_config, responses_registry, client=client, mcp_registry=StubMCPRegistry(mcp))
    tool = MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object", "properties": {}})
    with (
        patch("data_designer.engine.mcp.io.list_tools", return_value=[tool]),
        patch(
            "data_designer.engine.mcp.io.call_tools",
            return_value=[MCPToolResult("result-a"), MCPToolResult("result-b")],
        ),
    ):
        if asynchronous:
            result, trace = await facade.agenerate(prompt="question", parser=str, tool_alias="tools")
            await client.aclose()
        else:
            result, trace = facade.generate(prompt="question", parser=str, tool_alias="tools")
            client.close()
    assert result == "done"
    assert len([message for message in trace if message.role == "tool"]) == 4
    bodies = [json.loads(request.content) for request in httpx_mock.get_requests()]
    for body in bodies:
        assert body["store"] is False
        assert body["include"] == ["reasoning.encrypted_content"]
        assert body["reasoning"] == {"effort": "medium"}
        assert body["max_output_tokens"] == 1024
        assert body["tools"][0]["strict"] is False
    previous = bodies[0]["input"]
    for turn in range(2):
        current = bodies[turn + 1]["input"]
        assert current[: len(previous)] == previous
        assert current[len(previous) : len(previous) + 4] == rounds[turn]
        assert current[-2:] == [
            {
                "type": "function_call_output",
                "call_id": f"call_{turn}_{i}",
                "output": [{"type": "input_text", "text": value}],
            }
            for i, value in enumerate(["result-a", "result-b"])
        ]
        previous = current


def test_response_normalization_and_image_schema_request(
    httpx_mock: HTTPXMock, responses_client: OpenAIResponsesClient
) -> None:
    output = [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "summary"}], "encrypted_content": "secret"},
        make_text('{"score":1}'),
    ]
    httpx_mock.add_response(json=make_response(output))
    request = ChatCompletionRequest(
        api="responses",
        model="gpt-test",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Score"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
                ],
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "score", "strict": True, "schema": {"type": "object"}},
        },
        extra_body={"max_completion_tokens": 1024},
    )
    response = responses_client.completion(request)
    body = json.loads(httpx_mock.get_request().content)
    assert body["input"][0]["content"][1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,abc",
        "detail": "auto",
    }
    assert body["text"]["format"] == {
        "type": "json_schema",
        "name": "score",
        "strict": True,
        "schema": {"type": "object"},
    }
    assert response.message.content == '{"score":1}'
    assert response.message.reasoning_content == "summary"
    assert response.message.replay_data.payload == output
    assert response.usage.input_tokens == 10
    assert response.usage.reasoning_tokens == 3
    assert response.choices[0].finish_reason == "stop"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 2},
        {"stop": ["stop"]},
        {"extra_body": {"stream": True}},
        {"extra_body": {"store": True}},
        {"extra_body": {"previous_response_id": "resp_1"}},
        {"max_tokens": 10, "extra_body": {"max_output_tokens": 20}},
        {"extra_body": {"reasoning_effort": "medium", "reasoning": {"effort": "low"}}},
        {"tools": [{"type": "web_search"}]},
        {"extra_body": {"include": "reasoning.encrypted_content"}},
        {"messages": [{"role": "user", "content": [{"type": "audio"}]}]},
        {
            "messages": [{"role": "assistant", "content": "old"}],
            "message_replay_data": [ProviderReplayData("anthropic", [])],
        },
    ],
)
def test_rejects_unsupported_requests_before_http(
    kwargs: dict[str, Any], httpx_mock: HTTPXMock, responses_client: OpenAIResponsesClient
) -> None:
    request = ChatCompletionRequest(
        api="responses", **{"model": "gpt-test", "messages": [{"role": "user", "content": "Hi"}], **kwargs}
    )
    with pytest.raises(ProviderError) as exc:
        responses_client.completion(request)
    assert exc.value.kind == ProviderErrorKind.BAD_REQUEST
    assert not httpx_mock.get_requests()


@pytest.mark.parametrize(
    "body",
    [
        [],
        {"status": "failed", "error": {"code": "server_error"}},
        {"status": []},
        {"status": "completed", "output": ["invalid"]},
    ],
)
def test_normalizes_errors(httpx_mock: HTTPXMock, responses_client: OpenAIResponsesClient, body: Any) -> None:
    httpx_mock.add_response(json=body)
    with pytest.raises(ProviderError) as exc:
        responses_client.completion(ChatCompletionRequest(api="responses", model="gpt-test", messages=[]))
    assert exc.value.kind == ProviderErrorKind.API_ERROR


@pytest.mark.parametrize(
    "body,finish_reason,content",
    [
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [make_text("partial")],
            },
            "length",
            "partial",
        ),
        (
            {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}, "output": []},
            "content_filter",
            None,
        ),
        (
            {
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "Cannot comply"}]}],
            },
            "content_filter",
            None,
        ),
        ({"status": "completed", "output": [make_text("one"), make_text("two")]}, "stop", "onetwo"),
    ],
)
def test_normalizes_finish_reasons(
    httpx_mock: HTTPXMock,
    responses_client: OpenAIResponsesClient,
    body: dict[str, Any],
    finish_reason: str,
    content: str | None,
) -> None:
    httpx_mock.add_response(json=body)
    response = responses_client.completion(ChatCompletionRequest(api="responses", model="gpt-test", messages=[]))
    assert response.choices[0].finish_reason == finish_reason
    assert response.message.content == content
    assert response.usage is None


def test_responses_uses_anthropic_messages(httpx_mock: HTTPXMock, responses_config: ModelConfig) -> None:
    output = [{"type": "thinking", "thinking": "reasoning", "signature": "signed"}, {"type": "text", "text": "answer"}]
    httpx_mock.add_response(
        url="https://api.example.com/v1/messages", json={"content": output, "stop_reason": "end_turn"}
    )
    providers = ModelProviderRegistry(
        providers=[ModelProvider(name="responses", provider_type="anthropic", endpoint="https://api.example.com/v1")]
    )
    client = create_model_client(responses_config, PlaintextResolver(), providers)
    facade = ModelFacade(responses_config, providers, client=client)
    try:
        response = facade.responses([ChatMessage.as_user("question")])
        assert response.message.content == "answer"
        assert response.message.replay_data == ProviderReplayData("anthropic", output)
        assert "api" not in json.loads(httpx_mock.get_request().content)
    finally:
        facade.close()


def test_translates_canonical_history_and_forced_tool_without_mutation(
    httpx_mock: HTTPXMock, responses_client: OpenAIResponsesClient
) -> None:
    httpx_mock.add_response(json=make_response([make_text("done")]))
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}, "strict": True}}]
    request = ChatCompletionRequest(
        api="responses",
        model="gpt-test",
        messages=[
            {"role": "system", "content": "Be helpful"},
            {
                "role": "assistant",
                "content": "Checking",
                "tool_calls": [{"id": "call_1", "function": {"name": "lookup", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "lookup"}},
        n=1,
        extra_body={"include": ["reasoning.encrypted_content"], "reasoning": {"effort": "medium"}},
        extra_headers={"api-key": "test-key"},
        timeout=12,
    )
    responses_client.completion(request)
    sent = httpx_mock.get_request()
    body = json.loads(sent.content)
    assert body["tool_choice"] == {"type": "function", "name": "lookup"}
    assert body["tools"] == [{"type": "function", **tools[0]["function"]}]
    assert body["input"] == [
        {"role": "system", "content": [{"type": "input_text", "text": "Be helpful"}]},
        {"role": "assistant", "content": "Checking"},
        {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "result"},
    ]
    assert request.tools == tools
    assert request.extra_body == {"include": ["reasoning.encrypted_content"], "reasoning": {"effort": "medium"}}
    assert sent.headers["api-key"] == "test-key"
    assert sent.extensions["timeout"]["read"] == 12
    assert "n" not in body


@pytest.mark.parametrize("status,call_status", [("incomplete", "completed"), ("completed", "incomplete")])
def test_incomplete_function_calls_are_not_executed(
    httpx_mock: HTTPXMock, responses_client: OpenAIResponsesClient, status: str, call_status: str
) -> None:
    httpx_mock.add_response(
        json={
            "status": status,
            "output": [
                {
                    "type": "function_call",
                    "status": call_status,
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": "{}",
                }
            ],
        }
    )
    with pytest.raises(ProviderError, match="Incomplete Responses function call"):
        responses_client.completion(ChatCompletionRequest(api="responses", model="gpt-test", messages=[]))


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("api", ["chat-completions", "responses"])
async def test_explicit_methods_and_generation_use_one_provider(
    httpx_mock: HTTPXMock,
    responses_registry: ModelProviderRegistry,
    responses_config: ModelConfig,
    asynchronous: bool,
    api: str,
) -> None:
    httpx_mock.add_response(
        url="https://api.example.com/v1/chat/completions",
        json={"choices": [{"message": {"content": "chat"}}]},
        is_reusable=True,
    )
    httpx_mock.add_response(
        url="https://api.example.com/v1/responses", json=make_response([make_text("responses")]), is_reusable=True
    )
    config = responses_config.model_copy(update={"api": api})
    registry = create_model_registry(
        model_configs=[config],
        secret_resolver=PlaintextResolver(),
        model_provider_registry=responses_registry,
        client_concurrency_mode=ClientConcurrencyMode.ASYNC if asynchronous else ClientConcurrencyMode.SYNC,
    )
    model = registry.get_models(["responses"])["responses"]
    messages = [ChatMessage.as_user("question")]
    try:
        if asynchronous:
            chat = await model.acompletion(messages)
            response = await model.aresponses(messages)
            generated, _ = await model.agenerate("question")
        else:
            chat = await asyncio.to_thread(model.completion, messages)
            response = await asyncio.to_thread(model.responses, messages)
            generated, _ = await asyncio.to_thread(model.generate, "question")
        assert chat.message.content == "chat"
        assert response.message.content == "responses"
        assert generated == ("responses" if api == "responses" else "chat")
        assert model.usage_stats.request_usage.successful_requests == 3
        assert all("api" not in json.loads(request.content) for request in httpx_mock.get_requests())
    finally:
        if asynchronous:
            await model.aclose()
        else:
            model.close()
