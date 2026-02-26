# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    generated_images: int | None = None


@dataclass
class ImagePayload:
    # Canonical output shape to upper layers is base64 without data URI prefix.
    b64_data: str
    mime_type: str | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments_json: str


@dataclass
class AssistantMessage:
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    images: list[ImagePayload] = field(default_factory=list)


@dataclass
class ChatCompletionRequest:
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None
    extra_body: dict[str, Any] | None = None
    extra_headers: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ChatCompletionResponse:
    message: AssistantMessage
    usage: Usage | None = None
    raw: Any | None = None


@dataclass
class EmbeddingRequest:
    model: str
    inputs: list[str]
    encoding_format: str | None = None
    dimensions: int | None = None
    timeout: float | None = None
    extra_body: dict[str, Any] | None = None
    extra_headers: dict[str, str] | None = None


@dataclass
class EmbeddingResponse:
    vectors: list[list[float]]
    usage: Usage | None = None
    raw: Any | None = None


@dataclass
class ImageGenerationRequest:
    model: str
    prompt: str
    messages: list[dict[str, Any]] | None = None
    n: int | None = None
    timeout: float | None = None
    extra_body: dict[str, Any] | None = None
    extra_headers: dict[str, str] | None = None


@dataclass
class ImageGenerationResponse:
    images: list[ImagePayload]
    usage: Usage | None = None
    raw: Any | None = None
