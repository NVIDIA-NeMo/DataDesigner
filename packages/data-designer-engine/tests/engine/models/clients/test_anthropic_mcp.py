# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from data_designer.config.mcp import ToolConfig
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.engine.mcp import io as mcp_io
from data_designer.engine.mcp.facade import DEFAULT_TOOL_REFUSAL_MESSAGE, MCPFacade
from data_designer.engine.mcp.registry import MCPRegistry, MCPToolDefinition, MCPToolResult
from data_designer.engine.model_provider import MCPProviderRegistry, ModelProviderRegistry
from data_designer.engine.models.clients.adapters.anthropic import AnthropicClient
from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.models.parsers.errors import ParserException


def _parse_done(text: str) -> str:
    if text != "done":
        raise ParserException("Expected done")
    return text


@pytest.fixture
def anthropic_tool_rounds() -> list[list[dict[str, Any]]]:
    return [
        [
            {"type": "thinking", "thinking": "", "signature": f"empty-signature-{turn}"},
            {"type": "text", "text": "  Looking up.  "},
            {"type": "tool_use", "id": f"call-{turn}-a", "name": "lookup", "input": {"query": "a"}},
            {"type": "thinking", "thinking": "  Check another source.\n", "signature": f"signature-{turn}"},
            {"type": "redacted_thinking", "data": f"redacted-data-{turn}"},
            {"type": "tool_use", "id": f"call-{turn}-b", "name": "lookup", "input": {"query": "b"}},
            {"type": "text", "text": "  Waiting for results.\n"},
        ]
        for turn in range(2)
    ]


@pytest.fixture
def mcp_calls(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    monkeypatch.setattr(
        mcp_io,
        "list_tools",
        MagicMock(
            return_value=(MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),)
        ),
    )
    calls = MagicMock(return_value=[MCPToolResult(content="Result a"), MCPToolResult(content="Result b")])
    monkeypatch.setattr(mcp_io, "call_tools", calls)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("tool_turn_limit", [1, 2], ids=["refusal", "execution"])
@pytest.mark.parametrize("retry", ["none", "correction", "restart"])
async def test_generate_replays_anthropic_content(
    is_async: bool,
    tool_turn_limit: int,
    retry: str,
    anthropic_tool_rounds: list[list[dict[str, Any]]],
    mcp_calls: MagicMock,
    stub_tool_config: ToolConfig,
    stub_secret_resolver: MagicMock,
    stub_mcp_provider_registry: MCPProviderRegistry,
) -> None:
    stub_tool_config.max_tool_call_turns = tool_turn_limit
    mcp_registry = MCPRegistry(
        tool_configs=[stub_tool_config],
        secret_resolver=stub_secret_resolver,
        mcp_provider_registry=stub_mcp_provider_registry,
        mcp_facade_factory=MCPFacade,
    )
    invalid_content = [
        {"type": "thinking", "thinking": "", "signature": "final-signature"},
        {"type": "redacted_thinking", "data": "final-redacted"},
        {"type": "text", "text": "  invalid  "},
    ]
    responses = [{"content": blocks, "stop_reason": "tool_use"} for blocks in anthropic_tool_rounds]
    if retry != "none":
        responses.append({"content": invalid_content, "stop_reason": "end_turn"})
    responses.append({"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"})
    payloads: list[dict[str, Any]] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=responses[len(payloads) - 1])

    provider = ModelProvider(name="anthropic", endpoint="https://anthropic.test/v1", provider_type="anthropic")
    transport = httpx.MockTransport(handle_request)
    async with httpx.AsyncClient(transport=transport) as async_client:
        with httpx.Client(transport=transport) as sync_client:
            client = AnthropicClient(
                provider_name=provider.name,
                endpoint=provider.endpoint,
                concurrency_mode=ClientConcurrencyMode.ASYNC if is_async else ClientConcurrencyMode.SYNC,
                async_client=async_client if is_async else None,
                sync_client=sync_client if not is_async else None,
            )
            model = ModelFacade(
                model_config=ModelConfig(alias="claude", model="claude-test", provider=provider.name),
                model_provider_registry=ModelProviderRegistry(providers=[provider]),
                client=client,
                mcp_registry=mcp_registry,
            )
            kwargs = {
                "parser": _parse_done,
                "tool_alias": stub_tool_config.tool_alias,
                "max_correction_steps": int(retry == "correction"),
                "max_conversation_restarts": int(retry == "restart"),
            }
            if is_async:
                result, trace = await model.agenerate("Use the tools", **kwargs)
            else:
                result, trace = model.generate("Use the tools", **kwargs)

    assert result == "done"
    assert len(payloads) == len(responses)
    for request_index, payload in enumerate(payloads[1:], start=1):
        assistants = [message for message in payload["messages"] if message["role"] == "assistant"]
        assert [message["content"] for message in assistants[:2]] == anthropic_tool_rounds[:request_index]
        tool_results = [
            block
            for message in payload["messages"]
            if message["role"] == "user" and isinstance(message["content"], list)
            for block in message["content"]
            if block["type"] == "tool_result"
        ]
        expected_ids = [f"call-{turn}-{suffix}" for turn in range(min(request_index, 2)) for suffix in ("a", "b")]
        assert [block["tool_use_id"] for block in tool_results] == expected_ids
        for block in tool_results:
            expected = (
                DEFAULT_TOOL_REFUSAL_MESSAGE
                if block["tool_use_id"].startswith("call-1") and tool_turn_limit == 1
                else f"Result {block['tool_use_id'][-1]}"
            )
            assert block["content"] == expected

    if retry == "correction":
        assert payloads[-1]["messages"][-2] == {"role": "assistant", "content": invalid_content}
    elif retry == "restart":
        assert payloads[-1]["messages"] == payloads[-2]["messages"]
    assert mcp_calls.call_count == tool_turn_limit
    for call in mcp_calls.call_args_list:
        assert [(name, args) for _, name, args in call.args[0]] == [
            ("lookup", {"query": "a"}),
            ("lookup", {"query": "b"}),
        ]
    trace_json = json.dumps([message.to_dict() for message in trace])
    assert "signature" not in trace_json
    assert "redacted" not in trace_json
    assert "anthropic_content" not in trace_json
