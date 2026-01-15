# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from lxml.etree import _Element
from pydantic import BaseModel, Field
from typing_extensions import Self


class LLMStructuredResponse(BaseModel):
    """Output format for the LLM Response Parser."""

    response: str = Field(description="Raw Markdown/Markup response received from the LLM and input to the parser.")
    markup: str = Field(description="Markup/HTML resulting from running Markdown parsing on response.")
    parsed: list[BaseModel] = Field(
        default_factory=list,
        description="Structured content parsed from markup. Elements of this list are in document-order.",
    )

    def head(self, n: int) -> Self:
        """Retain only the first n elements of the parsed response."""
        out = self.model_copy()
        out.parsed = out.parsed[:n]
        return out

    def tail(self, n: int) -> Self:
        """Retain only the last n elements of the parsed response."""
        out = self.model_copy()
        out.parsed = out.parsed[-n:]
        return out

    def filter(self, block_types: list[type[BaseModel]]) -> Self:
        out = self.model_copy()
        out.parsed = [b for b in out.parsed if isinstance(b, tuple(block_types))]
        return out


@runtime_checkable
class TagParser(Protocol):
    """Protocol for tag parsing implementations.

    All TagParsers are objects which can take as input an `lxml`
    element, do some computation, and return some kind of structured
    output, represented as a subclass of Pydantic `BaseModel`.
    This protocol implementation can cover both classes as well
    as curried functions as parsers (e.g. `partial`).
    """

    def __call__(self, element: _Element) -> BaseModel: ...


@runtime_checkable
class PostProcessor(Protocol):
    """Protocol for parsed output postprocessing implementations.

    Implementations of this protocol are used to transform the results of
    the LLM response parser while retaining the same output structure.
    This is done so that PostProcessor implementations can be chained
    together.
    """

    def __call__(self, structured_response: LLMStructuredResponse) -> LLMStructuredResponse: ...


class TextBlock(BaseModel):
    text: str


class CodeBlock(BaseModel):
    code: str
    code_lang: str | None = None


class StructuredDataBlock(BaseModel):
    serialized: str
    obj: Any


class PydanticTypeBlock(BaseModel):
    serialized: str
    obj: BaseModel
