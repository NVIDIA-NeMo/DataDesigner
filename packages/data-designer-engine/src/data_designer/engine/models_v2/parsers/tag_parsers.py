# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from lxml.etree import _Element

from data_designer.engine.models.parsers.types import CodeBlock, TextBlock


def text_parser(element: _Element) -> TextBlock:
    return TextBlock(text=element.text if element.text else "")


def text_parser_keep_markup(element: _Element) -> TextBlock:
    body = element.text if element.text else ""
    return TextBlock(text=f"<{element.tag}>{body}</{element.tag}>")


def inline_code_parser(element: _Element) -> TextBlock:
    return TextBlock(text=f"`{element.text if element.text else ''}`")


def code_block_parser(element: _Element) -> CodeBlock:
    """Parse a <pre><code> element node.

    This parser handles the special case of Markdown->HTML conversion
    for fenced code blocks. These take on the form:

       ```xx
       ...
       ```

        <pre><code class="language-xx">...</code></pre>

    This parser is intended to be attached to the special case of "pre.code"
    tag hierarchies.

    Syntax Handling

        If the syntax is not specified, e.g. ``<code>...</code>`` or
        ``<code class="">...</code>``, then the syntax field is returned
        as None. However, the parser does not _enforce_ the prefix
        `language-` on the value of the class attribute.
        If it is not present, then the entire value

    Args:
        element (lxml.etree._Element): An element of the lxml-parsed
            element tree.

    Returns:
        CodeBlock: Datat structured containing both the body of the code
            as well as the specified synax of the code block.

    """
    prefix = "language-"
    language_identifier = element.attrib.get("class", "")
    language_identifier = language_identifier.removeprefix(prefix)
    return CodeBlock(
        code=element.text.strip() if element.text else "",
        code_lang=language_identifier if language_identifier else None,
    )
