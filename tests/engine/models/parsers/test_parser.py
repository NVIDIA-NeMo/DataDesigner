# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import BaseModel

from data_designer.engine.models.parsers.parser import LLMResponseParser
from data_designer.engine.models.parsers.postprocessors import (
    RealizePydanticTypes,
    deserialize_json_code,
    merge_text_blocks,
)
from data_designer.engine.models.parsers.tag_parsers import code_block_parser
from data_designer.engine.models.parsers.types import (
    CodeBlock,
    LLMStructuredResponse,
    PydanticTypeBlock,
    StructuredDataBlock,
    TextBlock,
)


def test_llm_response_parser_default():
    text = """\
Test prompt return. The return has some `code` included.

```python
import numpy as np
```
That is all there is at the moment.\
"""
    parser = LLMResponseParser(postprocessors=[])
    result = parser.parse(text)

    assert isinstance(result, LLMStructuredResponse)
    assert result.response == text
    assert result.markup  # existence


def test_llm_response_parser_merged():
    text = """\
Test prompt return. The return has some `code` included.
```python
import numpy as np
```
That is all there is at the moment.\
"""
    parser = LLMResponseParser(postprocessors=[merge_text_blocks])
    result = parser.parse(text)

    assert isinstance(result, LLMStructuredResponse)
    assert result.response == text
    assert result.markup  # existence
    assert len(result.parsed) == 3

    assert isinstance(result.parsed[0], TextBlock)
    assert result.parsed[0].text.replace("\n", "") == "Test prompt return. The return has some `code` included."

    assert isinstance(result.parsed[1], CodeBlock)
    assert result.parsed[1].code == "import numpy as np"
    assert result.parsed[1].code_lang == "python"

    assert isinstance(result.parsed[2], TextBlock)
    assert result.parsed[2].text.replace("\n", "") == "That is all there is at the moment."


def test_llm_response_parser_markup_passthrough():
    text = """\
<thinking>I'm thinking real hard</thinking><response>42!</response>\
"""
    parser = LLMResponseParser(postprocessors=[merge_text_blocks])
    result = parser.parse(text)

    assert isinstance(result, LLMStructuredResponse)
    assert result.response == text
    assert result.markup == f"<p>{text}</p>\n"
    assert len(result.parsed) == 1

    block = result.parsed[0]
    assert isinstance(block, TextBlock)
    assert block.text == text


def test_llm_response_parser_full_pipeline():
    text = """\
Test prompt return. The return has some `code` included.
```json
{"asdf": 42}
```
```json
{"baz": 3}
```
```json
{"foos": [{"baz": 1}, {"baz": 2}]}
```
That is all there is at the moment.\
"""

    class Foo(BaseModel):
        baz: int

    class Bar(BaseModel):
        foos: list[Foo]

    parser = LLMResponseParser(
        postprocessors=[
            merge_text_blocks,
            deserialize_json_code,
            RealizePydanticTypes([Foo, Bar]),
        ]
    )
    result = parser.parse(text)

    assert isinstance(result, LLMStructuredResponse)
    assert result.response == text
    assert result.markup  # existence
    assert len(result.parsed) == 5

    assert result.parsed[0] == TextBlock(text="Test prompt return. The return has some `code` included.")
    assert isinstance(result.parsed[1], StructuredDataBlock)
    assert isinstance(result.parsed[2], PydanticTypeBlock)
    assert isinstance(result.parsed[3], PydanticTypeBlock)

    assert result.parsed[1].obj == {"asdf": 42}
    assert result.parsed[2].obj == Foo(baz=3)
    assert result.parsed[3].obj == Bar(foos=[Foo(baz=1), Foo(baz=2)])
    assert result.parsed[4] == TextBlock(text="That is all there is at the moment.")


def test_llm_response_parser_custom_tag():
    text = """\
<thinking>
I had a thought and forgot it.
</thinking>

That is all there is at the moment.\
"""
    # Try default
    parser = LLMResponseParser(
        postprocessors=[
            merge_text_blocks,
        ]
    )

    result = parser.parse(text)
    assert len(result.parsed) == 1
    assert isinstance(result.parsed[0], TextBlock)

    # We should be preserving markdown tags in this case
    assert "<thinking>" in result.parsed[0].text
    assert "</thinking>" in result.parsed[0].text

    # Try with the parser engaged. Just
    # use a default one and see what happens.
    parser = LLMResponseParser(
        tag_parsers={"thinking": code_block_parser},
        postprocessors=[
            merge_text_blocks,
        ],
    )

    result = parser.parse(text)
    assert len(result.parsed) == 2
    assert isinstance(result.parsed[0], CodeBlock)
    assert result.parsed[0].code == "I had a thought and forgot it."


def test_llm_response_parser_empty_response():
    text = ""
    parser = LLMResponseParser(postprocessors=[merge_text_blocks])
    result = parser.parse(text)
    assert len(result.parsed) == 0
    assert result.response == text
    assert result.markup == ""
