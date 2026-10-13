# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import NoReturn

import json_repair

from data_designer.engine.models.parsers.types import (
    CodeBlock,
    LLMStructuredResponse,
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
    """Deserialize fenced JSON, falling back to a complete bare JSON response."""
    processed_response = structured_response.model_copy()
    processed_response.parsed = []

    for block in structured_response.parsed:
        if isinstance(block, CodeBlock) and block.code_lang == "json":
            deserialized = json_repair.loads(block.code)

            block = StructuredDataBlock(serialized=block.code, obj=deserialized)

            processed_response.parsed.append(block)
        else:
            processed_response.parsed.append(block)

    if not any(isinstance(block, StructuredDataBlock) for block in processed_response.parsed):
        try:
            deserialized = json.loads(structured_response.response, parse_constant=_reject_json_constant)
        except ValueError:
            return processed_response
        processed_response.parsed = [StructuredDataBlock(serialized=structured_response.response, obj=deserialized)]

    return processed_response


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"Non-standard JSON constant: {value}")
