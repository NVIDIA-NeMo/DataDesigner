# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from data_designer.engine.mcp.registry import MCPToolDefinition
from data_designer.engine.models.clients.adapters.anthropic_translation import (
    build_anthropic_payload,
    extract_system_content,
    merge_system_parts,
    parse_anthropic_response,
    parse_tool_call_arguments,
    translate_content_blocks,
    translate_image_url_block,
    translate_non_tool_message,
    translate_regular_content,
    translate_request_messages,
    translate_tool_call,
    translate_tool_calls,
    translate_tool_definition,
    translate_tool_result_content,
    translate_tool_result_message,
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


def test_build_anthropic_payload_preserves_multimodal_system_content() -> None:
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": "https://example.com/reference.png"},
                ],
            },
            {"role": "user", "content": "Go."},
        ],
    )

    payload = build_anthropic_payload(request)

    assert payload["system"] == [
        {"type": "text", "text": "Describe this image."},
        {"type": "image", "source": {"type": "url", "url": "https://example.com/reference.png"}},
    ]


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


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param("Be concise.", "Be concise.", id="string"),
        pytest.param(
            [
                {"type": "text", "text": "Rule 1"},
                {"type": "image_url", "image_url": "https://example.com/reference.png"},
                {"type": "text", "text": "Rule 2"},
            ],
            [
                {"type": "text", "text": "Rule 1"},
                {"type": "image", "source": {"type": "url", "url": "https://example.com/reference.png"}},
                {"type": "text", "text": "Rule 2"},
            ],
            id="mixed-text-and-image-returns-blocks",
        ),
        pytest.param(
            [{"type": "image_url", "image_url": "https://example.com/reference.png"}],
            [{"type": "image", "source": {"type": "url", "url": "https://example.com/reference.png"}}],
            id="image-only-returns-blocks",
        ),
        pytest.param(
            [{"type": "text", "text": "Rule 1"}, {"type": "text", "text": "Rule 2"}],
            "Rule 1\nRule 2",
            id="text-only-returns-string",
        ),
        pytest.param(None, None, id="none"),
        pytest.param("", None, id="empty-string"),
    ],
)
def test_extract_system_content_normalizes_supported_inputs(
    content: object, expected: str | list[dict[str, object]] | None
) -> None:
    assert extract_system_content(content) == expected


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        pytest.param(
            ["Part A", "Part B"],
            "Part A\n\nPart B",
            id="all-strings-joined",
        ),
        pytest.param(
            ["Part A"],
            "Part A",
            id="single-string",
        ),
        pytest.param(
            [
                "Text preamble",
                [
                    {"type": "text", "text": "Rule 1"},
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}},
                ],
            ],
            [
                {"type": "text", "text": "Text preamble"},
                {"type": "text", "text": "Rule 1"},
                {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}},
            ],
            id="mixed-string-and-blocks",
        ),
        pytest.param(
            [
                [{"type": "text", "text": "A"}],
                [{"type": "text", "text": "B"}],
            ],
            [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
            id="all-block-lists-flattened",
        ),
    ],
)
def test_merge_system_parts_normalizes_supported_inputs(
    parts: list[str | list[dict[str, object]]],
    expected: str | list[dict[str, object]],
) -> None:
    assert merge_system_parts(parts) == expected


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


def test_translate_non_tool_message_rejects_unsupported_role() -> None:
    with pytest.raises(ValueError, match="does not support message role"):
        translate_non_tool_message({"role": "system", "content": "Nope"})


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param("Hello", "Hello", id="string"),
        pytest.param(42, [{"type": "text", "text": "42"}], id="scalar"),
    ],
)
def test_translate_regular_content_normalizes_supported_inputs(
    content: object,
    expected: str | list[dict[str, object]],
) -> None:
    assert translate_regular_content(content) == expected


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        pytest.param(
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the knowledge base.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            },
            {
                "name": "search",
                "description": "Search the knowledge base.",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
            id="openai-shape",
        ),
        pytest.param(
            {
                "name": "search",
                "description": "Search the knowledge base.",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
            {
                "name": "search",
                "description": "Search the knowledge base.",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
            id="anthropic-shape",
        ),
    ],
)
def test_translate_tool_definition_normalizes_supported_shapes(
    tool: dict[str, object],
    expected: dict[str, object],
) -> None:
    assert translate_tool_definition(tool) == expected


def test_translate_content_blocks_converts_images_and_preserves_other_blocks() -> None:
    blocks = translate_content_blocks(
        [
            {"type": "image_url", "image_url": "https://example.com/cat.png"},
            {"type": "text", "text": "Caption"},
            {"type": "custom_block", "value": "kept"},
        ]
    )

    assert blocks == [
        {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}},
        {"type": "text", "text": "Caption"},
        {"type": "custom_block", "value": "kept"},
    ]


def test_translate_content_blocks_drops_malformed_image_url_block() -> None:
    blocks = translate_content_blocks(
        [
            {"type": "image_url"},
            {"type": "text", "text": "Kept"},
        ]
    )

    assert blocks == [{"type": "text", "text": "Kept"}]


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        pytest.param(
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR...", "format": "png"}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBOR..."}},
            id="data-uri",
        ),
        pytest.param(
            {"type": "image_url", "image_url": "https://example.com/cat.png"},
            {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}},
            id="remote-url",
        ),
    ],
)
def test_translate_image_url_block_normalizes_supported_inputs(
    block: dict[str, object],
    expected: dict[str, object],
) -> None:
    assert translate_image_url_block(block) == expected


@pytest.mark.parametrize(
    ("tool_call", "expected"),
    [
        pytest.param(
            {"type": "tool_use", "id": "call_1", "name": "search", "input": {"query": "weather"}},
            {"type": "tool_use", "id": "call_1", "name": "search", "input": {"query": "weather"}},
            id="anthropic-tool-use",
        ),
        pytest.param(
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query": "weather"}'},
            },
            {"type": "tool_use", "id": "call_1", "name": "search", "input": {"query": "weather"}},
            id="openai-function-tool-call",
        ),
        pytest.param(
            {"id": "call_1", "name": "search", "arguments_json": '{"query": "weather"}'},
            {"type": "tool_use", "id": "call_1", "name": "search", "input": {"query": "weather"}},
            id="canonical-tool-call",
        ),
    ],
)
def test_translate_tool_call_normalizes_supported_shapes(
    tool_call: dict[str, object],
    expected: dict[str, object],
) -> None:
    assert translate_tool_call(tool_call) == expected


@pytest.mark.parametrize(
    "tool_calls",
    [
        pytest.param({"id": "call_1"}, id="dict"),
        pytest.param("not-a-list", id="string"),
        pytest.param(None, id="none"),
    ],
)
def test_translate_tool_calls_rejects_non_list_input(tool_calls: object) -> None:
    with pytest.raises(ValueError, match="must be a list"):
        translate_tool_calls(tool_calls)


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        pytest.param(None, {}, id="none"),
        pytest.param({"query": "weather"}, {"query": "weather"}, id="dict"),
        pytest.param('{"query": "weather"}', {"query": "weather"}, id="json-string"),
    ],
)
def test_parse_tool_call_arguments_normalizes_supported_inputs(
    arguments: object,
    expected: dict[str, object],
) -> None:
    assert parse_tool_call_arguments(arguments) == expected


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        pytest.param('["not", "an", "object"]', "decode to a JSON object", id="json-array"),
        pytest.param("{not json}", "must be valid JSON", id="invalid-json"),
        pytest.param(123, "must be JSON or an object", id="wrong-type"),
    ],
)
def test_parse_tool_call_arguments_rejects_invalid_inputs(arguments: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_tool_call_arguments(arguments)


@pytest.mark.parametrize(
    "message",
    [
        pytest.param({"role": "tool", "content": "no id"}, id="missing-id"),
        pytest.param({"role": "tool", "tool_call_id": "", "content": "empty id"}, id="empty-id"),
    ],
)
def test_translate_tool_result_message_requires_tool_call_id(message: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="missing a tool_call_id"):
        translate_tool_result_message(message)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param("Line 1", "Line 1", id="string"),
        pytest.param(
            [{"type": "text", "text": "Line 1"}, {"type": "text", "text": "Line 2"}],
            "Line 1\nLine 2",
            id="text-blocks",
        ),
        pytest.param(
            [
                {"type": "image_url", "image_url": "https://example.com/chart.png"},
                {"type": "text", "text": "Caption"},
            ],
            [
                {"type": "image", "source": {"type": "url", "url": "https://example.com/chart.png"}},
                {"type": "text", "text": "Caption"},
            ],
            id="mixed-blocks",
        ),
    ],
)
def test_translate_tool_result_content_normalizes_supported_inputs(
    content: object,
    expected: str | list[dict[str, object]],
) -> None:
    assert translate_tool_result_content(content) == expected
