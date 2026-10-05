# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

import data_designer.engine.models.parsers.postprocessors as post
from data_designer.engine.models.parsers.types import (
    CodeBlock,
    LLMStructuredResponse,
    PostProcessor,
    StructuredDataBlock,
    TextBlock,
)

KNOWN_POSTPROCESSORS = [
    post.merge_text_blocks,
    post.deserialize_json_code,
]


@pytest.mark.parametrize("value", [{"answer": 42}, [1, 2], {}, [], 0, False, None, ""])
def test_deserialize_bare_json(value: object) -> None:
    raw = json.dumps(value)
    response = LLMStructuredResponse(response=raw, markup="", parsed=[TextBlock(text="rendered text")])

    result = post.deserialize_json_code(response)

    assert result.parsed == [StructuredDataBlock(serialized=raw, obj=value)]
    assert response.parsed == [TextBlock(text="rendered text")]


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_deserialize_bare_json_rejects_nonstandard_constants(constant: str) -> None:
    raw = f'{{"answer": {constant}}}'
    response = LLMStructuredResponse(response=raw, markup="", parsed=[TextBlock(text=raw)])
    assert post.deserialize_json_code(response).parsed == response.parsed


def test_protocol_adherence_postprocessors():
    for pp in KNOWN_POSTPROCESSORS:
        assert isinstance(pp, PostProcessor)


def test_merge_text_blocks():
    blocks = [
        TextBlock(text="a"),
        TextBlock(text="b"),
        TextBlock(text="c"),
        CodeBlock(code="a", code_lang="b"),
        TextBlock(text="c"),
        TextBlock(text="b"),
        TextBlock(text="a"),
    ]

    response = LLMStructuredResponse(response="", markup="", parsed=blocks)

    response = post.merge_text_blocks(response)
    assert len(response.parsed) == 3
    assert response.parsed[0] == TextBlock(text="abc")
    assert response.parsed[2] == TextBlock(text="cba")


def test_deserialize_json_code():
    blocks = [
        TextBlock(text="abc"),
        CodeBlock(code='{"foo": 42}', code_lang="json"),
        # And one with missing value
        CodeBlock(code='{"bar": 43, "baz": [1, 2, 3]', code_lang="json"),
        TextBlock(text="cba"),
    ]

    response = LLMStructuredResponse(response="", markup="", parsed=blocks)

    response = post.deserialize_json_code(response)
    assert len(response.parsed) == 4
    assert isinstance(response.parsed[1], StructuredDataBlock)
    assert response.parsed[1].obj == {"foo": 42}

    assert isinstance(response.parsed[2], StructuredDataBlock)
    assert response.parsed[2].obj == {"bar": 43, "baz": [1, 2, 3]}
