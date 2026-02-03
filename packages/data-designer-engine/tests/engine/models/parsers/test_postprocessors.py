# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

import data_designer.engine.models.parsers.postprocessors as post
from data_designer.engine.models.parsers.types import (
    CodeBlock,
    LLMStructuredResponse,
    PostProcessor,
    PydanticTypeBlock,
    StructuredDataBlock,
    TextBlock,
)

KNOWN_POSTPROCESSORS = [
    post.merge_text_blocks,
    post.deserialize_json_code,
    post.RealizePydanticTypes,
]


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


def test_realize_pydantic_types():
    class Foo(BaseModel):
        baz: int

    class Bar(BaseModel):
        foos: list[Foo]

    blocks = [
        TextBlock(text="abc"),
        StructuredDataBlock(serialized="", obj={"asdf": "q"}),
        StructuredDataBlock(serialized="", obj={"baz": 42}),
        StructuredDataBlock(serialized="", obj={"foos": [{"baz": 1}, {"baz": 2}]}),
        TextBlock(text="cba"),
    ]

    response = LLMStructuredResponse(response="", markup="", parsed=blocks)

    parser = post.RealizePydanticTypes([Foo, Bar])
    response = parser(response)

    assert len(response.parsed) == 5
    assert isinstance(response.parsed[1], StructuredDataBlock)
    assert isinstance(response.parsed[2], PydanticTypeBlock)
    assert isinstance(response.parsed[3], PydanticTypeBlock)

    assert response.parsed[2].obj == Foo(baz=42)
    assert response.parsed[3].obj == Bar(foos=[Foo(baz=1), Foo(baz=2)])


def test_deserialize_realize_pipeline():
    class Foo(BaseModel):
        baz: int

    class Bar(BaseModel):
        foos: list[Foo]

    blocks = [
        TextBlock(text="abc"),
        CodeBlock(code='{"asdf": "q"}', code_lang="json"),
        CodeBlock(code='{"baz": 42}', code_lang="json"),
        CodeBlock(code='{"foos": [{"baz": 1}, {"baz": 2}]}', code_lang="json"),
        TextBlock(text="cba"),
    ]

    response = LLMStructuredResponse(response="", markup="", parsed=blocks)

    parser = post.RealizePydanticTypes([Foo, Bar])
    response = parser(post.deserialize_json_code(response))

    assert len(response.parsed) == 5
    assert isinstance(response.parsed[1], StructuredDataBlock)
    assert isinstance(response.parsed[2], PydanticTypeBlock)
    assert isinstance(response.parsed[3], PydanticTypeBlock)

    assert response.parsed[2].obj == Foo(baz=42)
    assert response.parsed[3].obj == Bar(foos=[Foo(baz=1), Foo(baz=2)])
