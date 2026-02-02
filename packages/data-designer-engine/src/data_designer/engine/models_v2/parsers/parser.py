# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import reduce

import marko
from lxml import etree
from lxml.etree import _Element

import data_designer.engine.models.parsers.tag_parsers as tp
from data_designer.engine.models.parsers.postprocessors import merge_text_blocks
from data_designer.engine.models.parsers.types import (
    LLMStructuredResponse,
    PostProcessor,
    TagParser,
)

DEFAULT_TAG_PARSERS = {
    "pre.code": tp.code_block_parser,
    "p.code": tp.inline_code_parser,
    "p": tp.text_parser,
    "pre": tp.text_parser,
    "": tp.text_parser_keep_markup,
}

DEFAULT_POST_PROCESSORS = [merge_text_blocks]


def _patch_tags_before_code_fences(response: str) -> str:
    """Patch to add a linebreak between a tag prior to a code block.

    Marko conversion of MD->HTML has a quirk. If there is a case like
    the following, it will not convert the code block at all:

            ...
            </ending_tag>
            ```syntax
            ...

    We want to find these cases and simply introduce an additional
    line break.
    """

    return response.replace(">\n```", ">\n\n```")


class LLMResponseParser:
    """
    Parses Language Model (LLM) responses containing a mixture of Markdown and custom markup into structured data.

    The `LLMResponseParser` class facilitates the translation of LLM-generated responses, which may include
    Markdown and custom markup tags, into a structured format using ElementTree. It allows for customizable
    parsing behavior through the registration of tag-specific parsers and post-processors.

    ## Description

    The core functionality of this class enables LLMs to respond using Markdown along with any custom
    prompted markup specified by the system or task. The parsing process involves converting the Markdown
    and markup into an ElementTree, then processing each element using registered tag parsers to produce
    a list of structured `BaseModel` instances. Post-processors can further refine the structured response.

    ### Tag Parsers

    Tag parsers are responsible for handling specific markup tags within the LLM response. They can be
    registered with the parser using dot-path notation to manage hierarchical tag structures. This allows
    downstream tasks to customize how specific elements are processed into `BaseModel` instances.

    ### Post-Processors

    Post-processors are functions that operate on the list of parsed blocks to perform additional
    transformations or aggregations. They are applied after the initial parsing of the response.

    Attributes:
        tag_parsers (dict[str, TagParser]): A dictionary mapping tag paths to their corresponding `TagParser` instances.
        postprocessors (list[PostProcessor]): A list of post-processing functions to apply to the structured response.

    Example:
        ```python
        class CodeBlock(BaseModel):
            code: str
            syntax: Optional[str] = None

        class CodeBlockParser:
            def __call__(self, element: _Element) -> CodeBlock:
                # Implementation details...
                return CodeBlock(code=element.text, syntax=element.get("class"))

        parser = LLMResponseParser(
            tag_parsers={
                "pre.code": CodeBlockParser(),
            }
        )

        out = parser.parse('```json\n{"answer": 42}\n```')
        print(out.parsed)
        # Output: [CodeBlock(code='{"answer": 42}\n', syntax='json')]
        ```
    """

    tag_parsers: dict[str, TagParser]
    postprocessors: list[PostProcessor]

    def __init__(
        self,
        tag_parsers: dict[str, TagParser] | None = None,
        postprocessors: list[PostProcessor] | None = None,
    ):
        """
        Initializes the LLMResponseParser with optional tag parsers and post-processors.

        Args:
            tag_parsers (Optional[dict[str, TagParser]]): A dictionary mapping tag paths to `TagParser` instances.
                If provided, these parsers will be merged with the default tag parsers.
            postprocessors (Optional[list[PostProcessor]]): A list of post-processing functions to apply
                to the structured response. If not provided, a default post-processor `merge_text_blocks`
                is used.

        Attributes:
            tag_parsers (dict[str, TagParser]): Initialized with default tag parsers, updated with any provided.
            postprocessors (list[PostProcessor]): Initialized with default post-processors or the provided list.
        """
        self.tag_parsers = {**DEFAULT_TAG_PARSERS}
        if tag_parsers:
            self.tag_parsers.update(tag_parsers)

        self.postprocessors = [
            merge_text_blocks,
        ]
        if postprocessors is not None:
            self.postprocessors = postprocessors

    def lookup_parser(self, element: _Element) -> TagParser:
        """
        Resolves and retrieves the appropriate `TagParser` for a given XML element based on its tag hierarchy.

        The method constructs the dot-path lineage of the element's tags, starting from the root and moving
        towards the specific element. It then attempts to find the most specific matching `TagParser` by
        progressively reducing the specificity of the tag path until a matching parser is found.

        Args:
            element (_Element): The XML element for which to find the corresponding `TagParser`.

        Returns:
            TagParser: The `TagParser` instance that matches the element's tag path.

        Raises:
            KeyError: If no matching `TagParser` is found for the element's tag path.
        """
        # Get the dot path lineage of this tag, sans root.
        # Note that the lineage comes back in reverse order.
        parents = [e.tag for e in element.iterancestors()][::-1]
        lineage = [*parents, element.tag]

        # Now attempt to matchup with the tag parsers name.
        # Starts from the full linear (most specific), and
        # breaks on the first hit. So this should properly
        # prioritize specific parsers over general ones.
        while lineage:
            tag_path = ".".join(lineage)
            if tag_path not in self.tag_parsers:
                lineage.pop(0)
            else:
                break

        # Tag path can be an empty string, which hits the
        # default parsing option specified by the "" entry
        # of the tag parsers dict.
        tag_path = ".".join(lineage)
        return self.tag_parsers[tag_path]

    def postprocess(self, structured_response: LLMStructuredResponse) -> LLMStructuredResponse:
        """
        Applies post-processing functions to the structured response.

        If no post-processors are registered, the original structured response is returned.
        Otherwise, each post-processor is applied in sequence to transform the response.

        Args:
            structured_response (LLMStructuredResponse): The initial structured response to be post-processed.

        Returns:
            LLMStructuredResponse: The post-processed structured response.
        """
        if not self.postprocessors:
            return structured_response

        return reduce(lambda acc, func: func(acc), self.postprocessors, structured_response)

    def parse(self, md_response: str) -> LLMStructuredResponse:
        """
        Parses a Markdown-formatted LLM response into a structured `LLMStructuredResponse`.

        The parsing process involves converting the Markdown and custom markup into an XML tree,
        iterating over each element in a depth-first traversal to apply the appropriate
        `TagParser`, and then applying any registered post-processors to the resulting structured data.

        Args:
            md_response (str): The Markdown-formatted response from the LLM, potentially containing custom markup.

        Returns:
            LLMStructuredResponse: The structured representation of the parsed response, containing parsed blocks.

        Raises:
            etree.XMLSyntaxError: If the provided Markdown cannot be converted into a valid XML structure.
        """
        response = marko.convert(_patch_tags_before_code_fences(md_response))
        output = LLMStructuredResponse(response=md_response, markup=response)

        # Generate document tree
        parser = etree.HTMLParser(recover=True, remove_blank_text=True)
        root = etree.fromstring(response, parser=parser)
        tags = root.iter() if root is not None else []

        # Iterate over tags, depth first
        for element in tags:
            if element == root or element.tag == "body":
                continue

            parsed_block = self.lookup_parser(element)(element)

            # Make a quick check for dead text blocks, which
            # can happen with container tags like <pre>, <ul>, and <ol>.
            drop_block = isinstance(parsed_block, tp.TextBlock) and not parsed_block.text.strip()

            if not drop_block:
                output.parsed.append(parsed_block)

            # Check tails -- inelegant, but they're always text.
            # Don't add the tail if it is just blank space.
            if element.tail and element.tail.strip():
                output.parsed.append(tp.TextBlock(text=element.tail))

        return self.postprocess(output)
