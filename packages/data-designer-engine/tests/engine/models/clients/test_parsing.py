# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.engine.models.clients.parsing import extract_reasoning_content, extract_tool_calls
from data_designer.engine.models.clients.types import (
    ChatCompletionRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
    TransportKwargs,
)

# --- TransportKwargs.from_request: extra_body flattening ---


def test_extra_body_keys_are_flattened_into_body() -> None:
    request = ChatCompletionRequest(
        model="m",
        messages=[],
        temperature=0.7,
        extra_body={"reasoning_effort": "high", "seed": 42},
    )
    transport = TransportKwargs.from_request(request)

    assert transport.body["temperature"] == 0.7
    assert transport.body["reasoning_effort"] == "high"
    assert transport.body["seed"] == 42
    assert "extra_body" not in transport.body


def test_extra_body_none_produces_no_extra_keys() -> None:
    request = ChatCompletionRequest(model="m", messages=[], temperature=0.5)
    transport = TransportKwargs.from_request(request)

    assert transport.body == {"temperature": 0.5}
    assert "extra_body" not in transport.body


def test_extra_body_empty_dict_produces_no_extra_keys() -> None:
    request = ChatCompletionRequest(model="m", messages=[], extra_body={})
    transport = TransportKwargs.from_request(request)

    assert "extra_body" not in transport.body


# --- TransportKwargs.from_request: extra_headers separation ---


def test_extra_headers_are_separated_into_headers() -> None:
    request = ChatCompletionRequest(
        model="m",
        messages=[],
        extra_headers={"X-Custom": "value", "Authorization": "Bearer tok"},
    )
    transport = TransportKwargs.from_request(request)

    assert transport.headers == {"X-Custom": "value", "Authorization": "Bearer tok"}
    assert "extra_headers" not in transport.body


def test_extra_headers_none_produces_empty_headers() -> None:
    request = ChatCompletionRequest(model="m", messages=[])
    transport = TransportKwargs.from_request(request)

    assert transport.headers == {}


# --- TransportKwargs.from_request: combined ---


def test_extra_body_and_headers_together() -> None:
    request = ChatCompletionRequest(
        model="m",
        messages=[],
        temperature=0.9,
        max_tokens=100,
        extra_body={"seed": 1},
        extra_headers={"X-Req-Id": "abc"},
    )
    transport = TransportKwargs.from_request(request)

    assert transport.body == {"temperature": 0.9, "max_tokens": 100, "seed": 1}
    assert transport.headers == {"X-Req-Id": "abc"}


# --- TransportKwargs.from_request: exclude parameter ---


def test_exclude_removes_fields_from_body() -> None:
    request = ImageGenerationRequest(
        model="m",
        prompt="draw a cat",
        messages=[{"role": "user", "content": "hi"}],
        n=2,
        extra_body={"quality": "hd"},
    )
    transport = TransportKwargs.from_request(request, exclude=frozenset({"messages", "prompt"}))

    assert "messages" not in transport.body
    assert "prompt" not in transport.body
    assert transport.body["n"] == 2
    assert transport.body["quality"] == "hd"


# --- TransportKwargs.from_request: works with all request types ---


def test_embedding_request() -> None:
    request = EmbeddingRequest(
        model="m",
        inputs=["hello"],
        extra_body={"input_type": "query"},
        extra_headers={"X-Api-Version": "2"},
    )
    transport = TransportKwargs.from_request(request)

    assert transport.body["input_type"] == "query"
    assert transport.headers == {"X-Api-Version": "2"}
    assert "extra_body" not in transport.body
    assert "extra_headers" not in transport.body


def test_image_generation_request() -> None:
    request = ImageGenerationRequest(
        model="m",
        prompt="sunset",
        n=3,
        extra_body={"size": "1024x1024"},
    )
    transport = TransportKwargs.from_request(request)

    assert transport.body["n"] == 3
    assert transport.body["size"] == "1024x1024"
    assert transport.headers == {}


# --- TransportKwargs: falsy headers ---


def test_transport_kwargs_empty_headers_is_falsy() -> None:
    tk = TransportKwargs(body={"a": 1}, headers={})
    assert not tk.headers


@pytest.mark.parametrize(
    ("extra_body", "expected_body_keys"),
    [
        (None, set()),
        ({}, set()),
        ({"a": 1}, {"a"}),
        ({"a": 1, "b": 2}, {"a", "b"}),
    ],
)
def test_extra_body_variations(extra_body: dict | None, expected_body_keys: set[str]) -> None:
    request = ChatCompletionRequest(model="m", messages=[], extra_body=extra_body)
    transport = TransportKwargs.from_request(request)

    assert expected_body_keys.issubset(transport.body.keys())
    assert "extra_body" not in transport.body


# --- extract_tool_calls ---


def _make_raw_tool_call(
    tool_id: str | None = "call-1",
    name: str = "lookup",
    arguments: str = '{"q": "test"}',
) -> dict:
    tc: dict = {"type": "function", "function": {"name": name, "arguments": arguments}}
    if tool_id is not None:
        tc["id"] = tool_id
    return tc


def test_extract_tool_calls_basic() -> None:
    raw = [_make_raw_tool_call()]
    result = extract_tool_calls(raw)

    assert len(result) == 1
    assert result[0].id == "call-1"
    assert result[0].name == "lookup"
    assert result[0].arguments_json == '{"q": "test"}'


@pytest.mark.parametrize("tool_id", [None, ""], ids=["missing_id", "empty_string_id"])
def test_extract_tool_calls_falsy_id_generates_uuid(tool_id: str | None) -> None:
    raw = [_make_raw_tool_call(tool_id=tool_id)]
    result = extract_tool_calls(raw)

    assert len(result) == 1
    assert len(result[0].id) == 32  # uuid4().hex length
    assert result[0].id.isalnum()


def test_extract_tool_calls_multiple_missing_ids_are_unique() -> None:
    raw = [_make_raw_tool_call(tool_id=None), _make_raw_tool_call(tool_id=None)]
    result = extract_tool_calls(raw)

    assert result[0].id != result[1].id


@pytest.mark.parametrize("raw_input", [None, []], ids=["none", "empty_list"])
def test_extract_tool_calls_empty_input(raw_input: list | None) -> None:
    assert extract_tool_calls(raw_input) == []


def test_extract_tool_calls_none_arguments() -> None:
    raw = [{"id": "call-1", "function": {"name": "lookup", "arguments": None}}]
    result = extract_tool_calls(raw)

    assert result[0].arguments_json == "{}"


# --- extract_reasoning_content (vLLM field migration) ---


@pytest.mark.parametrize(
    "message,expected",
    [
        ({"reasoning": "step-by-step thinking"}, "step-by-step thinking"),
        ({"reasoning_content": "legacy thinking"}, "legacy thinking"),
        ({"reasoning": "canonical", "reasoning_content": "legacy"}, "canonical"),
        ({"content": "hello"}, None),
        (None, None),
        ({"reasoning": "", "reasoning_content": "fallback"}, "fallback"),
    ],
    ids=[
        "only-reasoning",
        "only-reasoning_content",
        "both-reasoning-takes-precedence",
        "neither-field",
        "none-message",
        "empty-reasoning-falls-back",
    ],
)
def test_extract_reasoning_content(message: dict | None, expected: str | None) -> None:
    assert extract_reasoning_content(message) == expected


def test_extract_reasoning_content_works_with_object_style_message() -> None:
    class Msg:
        reasoning = "from object"
        reasoning_content = "legacy object"

    assert extract_reasoning_content(Msg()) == "from object"
