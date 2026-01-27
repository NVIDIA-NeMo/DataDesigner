# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import reduce

from lxml.etree import _Element
from pydantic import BaseModel

from data_designer.engine.models.parsers.types import (
    CodeBlock,
    LLMStructuredResponse,
    PostProcessor,
    TagParser,
    TextBlock,
)


def test_llm_structured_response():
    response = LLMStructuredResponse(response="Hello", markup="<p>Hello</p>")
    assert response.parsed == []


def test_llm_structured_response_chaining():
    response = LLMStructuredResponse(
        response="",
        markup="",
        parsed=[
            TextBlock(text="a"),
            TextBlock(text="b"),
            TextBlock(text="c"),
            TextBlock(text="d"),
            CodeBlock(code="test_a()", code_lang=None),
            CodeBlock(code="test_b()", code_lang=None),
        ],
    )

    assert response.head(2).parsed == [TextBlock(text="a"), TextBlock(text="b")]
    assert response.head(2).tail(1).parsed == [TextBlock(text="b")]
    assert response.filter([CodeBlock]).head(1).parsed == [CodeBlock(code="test_a()", code_lang=None)]


def test_protocol_tag_parser():
    class Foo(BaseModel):
        result: int

    def parse_fn(element: _Element) -> Foo:
        return Foo(result=1)

    class Parse:
        def __call__(self, element: _Element) -> Foo:
            return Foo(result=1)

    assert isinstance(parse_fn, TagParser)
    assert isinstance(Parse, TagParser)


def test_protocol_post_processor():
    def post_fn(structured_response: LLMStructuredResponse) -> LLMStructuredResponse:
        return structured_response

    class PostClass:
        def __call__(self, structured_response: LLMStructuredResponse) -> LLMStructuredResponse:
            return structured_response

    assert isinstance(post_fn, PostProcessor)
    assert isinstance(PostClass, PostProcessor)


def test_protocol_post_processor_chaining():
    class Foo(BaseModel):
        result: int

    def repeat_block(
        structured_response: LLMStructuredResponse,
    ) -> LLMStructuredResponse:
        out = structured_response.model_copy()
        out.parsed = [*out.parsed, out.parsed[-1]]
        return out

    assert isinstance(repeat_block, PostProcessor)

    block = Foo(result=1)
    init_response = LLMStructuredResponse(
        response="",
        markup="",
        parsed=[
            block,
        ],
    )
    n_repeats = 5

    post_processors = [repeat_block for _ in range(n_repeats)]
    final_result = reduce(lambda acc, func: func(acc), post_processors, init_response)

    assert len(final_result.parsed) == n_repeats + 1
    assert all([b == block for b in final_result.parsed])
