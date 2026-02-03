# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json_repair
from pydantic import BaseModel, ValidationError

from data_designer.engine.models.parsers.types import (
    CodeBlock,
    LLMStructuredResponse,
    PydanticTypeBlock,
    StructuredDataBlock,
    TextBlock,
)


def merge_text_blocks(
    structured_response: LLMStructuredResponse,
) -> LLMStructuredResponse:
    processed_response = structured_response.model_copy()
    processed_response.parsed = []
    accumulator = None
    for block in structured_response.parsed:
        if isinstance(block, TextBlock):
            if accumulator is not None:
                accumulator = TextBlock(text=accumulator.text + block.text)
            else:
                accumulator = block
        else:
            if accumulator is not None:
                processed_response.parsed.append(accumulator)
                accumulator = None

            processed_response.parsed.append(block)

    if accumulator:
        processed_response.parsed.append(accumulator)

    return processed_response


def deserialize_json_code(
    structured_response: LLMStructuredResponse,
) -> LLMStructuredResponse:
    processed_response = structured_response.model_copy()
    processed_response.parsed = []

    for block in structured_response.parsed:
        if isinstance(block, CodeBlock) and block.code_lang == "json":
            deserialized = json_repair.loads(block.code)

            block = StructuredDataBlock(serialized=block.code, obj=deserialized)

            processed_response.parsed.append(block)
        else:
            processed_response.parsed.append(block)

    return processed_response


class RealizePydanticTypes:
    types: list[type[BaseModel]]

    def __init__(self, types: list[type[BaseModel]]):
        self.types = types

    def _fit_types(self, obj: dict) -> BaseModel | None:
        final_obj = None

        for t in self.types:
            try:
                final_obj = t.model_validate(obj)
            except ValidationError:
                pass

        return final_obj

    def __call__(self, structured_response: LLMStructuredResponse) -> LLMStructuredResponse:
        processed_response = structured_response.model_copy()
        processed_response.parsed = []

        for block in structured_response.parsed:
            if isinstance(block, StructuredDataBlock):
                new_block = block
                pydantic_obj = self._fit_types(block.obj)
                if pydantic_obj:
                    new_block = PydanticTypeBlock(serialized=block.serialized, obj=pydantic_obj)
                processed_response.parsed.append(new_block)
            else:
                processed_response.parsed.append(block)

        return processed_response
