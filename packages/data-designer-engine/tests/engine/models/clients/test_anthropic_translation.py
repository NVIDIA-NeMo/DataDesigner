# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from data_designer.engine.mcp.registry import MCPToolDefinition
from data_designer.engine.models.clients.adapters.anthropic_translation import (
    build_anthropic_payload,
    parse_anthropic_response,
    parse_tool_call_arguments,
    translate_image_url_block,
    translate_request_messages,
    translate_tool_definition,
)
from data_designer.engine.models.clients.types import ChatCompletionRequest
from data_designer.engine.models.utils import ChatMessage

MODEL = "claude-test"


def test_build_anthropic_payload_extracts_system_from_normalized_messages() -> None:
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            ChatMessage.as_system("Be concise.").to_dict(),
            ChatMessage.as_user("Hi").to_dict(),
        ],
    )

    payload = build_anthropic_payload(request)

    assert payload["system"] == "Be concise."
    assert payload["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    assert payload["max_tokens"] == 4096


def test_build_anthropic_payload_translates_tool_schema_and_turns() -> None:
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            ChatMessage.as_user("What's the weather?").to_dict(),
            ChatMessage.as_assistant(
                content="Let me check.",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"query": "weather"}'},
                    }
                ],
            ).to_dict(),
            ChatMessage.as_tool(content="Sunny and 72F", tool_call_id="call_1").to_dict(),
        ],
        tools=[
            MCPToolDefinition(
                name="search",
                description="Search the knowledge base.",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            ).to_openai_tool_schema()
        ],
    )

    payload = build_anthropic_payload(request)

    assert payload["tools"] == [
        {
            "name": "search",
            "description": "Search the knowledge base.",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]
    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "What's the weather?"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "call_1", "name": "search", "input": {"query": "weather"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "Sunny and 72F"}],
        },
    ]


def test_translate_request_messages_merges_parallel_tool_results() -> None:
    system_parts, messages = translate_request_messages(
        [
            ChatMessage.as_system("Be helpful.").to_dict(),
            ChatMessage.as_user("Plan my day.").to_dict(),
            ChatMessage.as_assistant(
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup_city", "arguments": '{"city": "Paris"}'},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "lookup_weather", "arguments": '{"city": "Paris"}'},
                    },
                ]
            ).to_dict(),
            ChatMessage.as_tool(content="City found", tool_call_id="call_1").to_dict(),
            ChatMessage.as_tool(content="Sunny", tool_call_id="call_2").to_dict(),
        ]
    )

    assert system_parts == ["Be helpful."]
    assert messages[-1] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "City found"},
            {"type": "tool_result", "tool_use_id": "call_2", "content": "Sunny"},
        ],
    }


def test_parse_anthropic_response_maps_tool_use_and_thinking() -> None:
    response = parse_anthropic_response(
        {
            "content": [
                {"type": "thinking", "thinking": "Let me reason."},
                {"type": "text", "text": "The answer is 42."},
                {"type": "tool_use", "id": "toolu_01", "name": "search", "input": {"query": "weather"}},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )

    assert response.message.content == "The answer is 42."
    assert response.message.reasoning_content == "Let me reason."
    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert json.loads(response.message.tool_calls[0].arguments_json) == {"query": "weather"}


def test_translate_tool_definition_converts_openai_shape() -> None:
    tool = translate_tool_definition(
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the knowledge base.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    )

    assert tool == {
        "name": "search",
        "description": "Search the knowledge base.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
    }


def test_translate_image_url_block_converts_data_uri() -> None:
    block = translate_image_url_block(
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR...", "format": "png"}}
    )

    assert block == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "iVBOR..."},
    }


def test_parse_tool_call_arguments_rejects_non_object_json() -> None:
    with pytest.raises(ValueError, match="decode to a JSON object"):
        parse_tool_call_arguments('["not", "an", "object"]')
