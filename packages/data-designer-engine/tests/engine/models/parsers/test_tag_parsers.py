# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from lxml.etree import Element

from data_designer.engine.models.parsers import tag_parsers as tp
from data_designer.engine.models.parsers.types import CodeBlock, TagParser, TextBlock

KNOWN_TAG_PARSERS = [
    tp.text_parser,
    tp.text_parser_keep_markup,
    tp.inline_code_parser,
    tp.code_block_parser,
]


@pytest.mark.parametrize("parser", KNOWN_TAG_PARSERS)
def test_protocol_adherence(parser):
    assert isinstance(parser, TagParser)


def test_text_parser():
    element = Element("tag")
    element.text = "Hello world!\nHello world!\n"
    parsed = tp.text_parser(element)

    assert isinstance(parsed, TextBlock)
    assert parsed.text == element.text


def test_text_parser_textless():
    element = Element("tag")
    parsed = tp.text_parser(element)

    assert isinstance(parsed, TextBlock)
    assert parsed.text == ""


def test_text_parser_keep_markup():
    element = Element("tag")
    element.text = "Hello world!\nHello world!\n"
    parsed = tp.text_parser_keep_markup(element)

    assert isinstance(parsed, TextBlock)
    assert parsed.text == "<tag>Hello world!\nHello world!\n</tag>"


def test_inline_code_parser():
    element = Element("code")
    element.text = "variable"

    parsed = tp.inline_code_parser(element)
    assert isinstance(parsed, TextBlock)
    assert parsed.text == "`variable`"


def test_inline_code_parser_null_text():
    element = Element("code")

    parsed = tp.inline_code_parser(element)
    assert isinstance(parsed, TextBlock)
    assert parsed.text == "``"


def test_code_block_parser():
    element = Element("code")
    element.set("class", "language-python")
    element.text = "import nump as np\n print('hello world!')"

    parsed = tp.code_block_parser(element)

    assert isinstance(parsed, CodeBlock)
    assert parsed.code == element.text
    assert parsed.code_lang == "python"

    element.set("class", "asdf")

    parsed = tp.code_block_parser(element)

    assert isinstance(parsed, CodeBlock)
    assert parsed.code == element.text
    assert parsed.code_lang == "asdf"


def test_code_block_parser_null_language():
    element = Element("code")
    element.text = "import nump as np\n print('hello world!')"

    parsed = tp.code_block_parser(element)

    assert isinstance(parsed, CodeBlock)
    assert parsed.code == element.text
    assert parsed.code_lang is None

    element.set("class", "language-")
    parsed = tp.code_block_parser(element)

    assert isinstance(parsed, CodeBlock)
    assert parsed.code == element.text
    assert parsed.code_lang is None

    element.set("class", "")
    parsed = tp.code_block_parser(element)

    assert isinstance(parsed, CodeBlock)
    assert parsed.code == element.text
    assert parsed.code_lang is None


def test_code_block_parser_null_text():
    element = Element("code")

    parsed = tp.code_block_parser(element)

    assert isinstance(parsed, CodeBlock)
    assert parsed.code == ""
    assert parsed.code_lang is None
