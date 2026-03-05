# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Protocol


class HttpResponse(Protocol):
    """Structural type for HTTP response objects (httpx, requests, etc.)."""

    status_code: int
    text: str

    def json(self) -> Any: ...


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


# ---------------------------------------------------------------------------
# Transport preparation
# ---------------------------------------------------------------------------


@dataclass
class TransportKwargs:
    """Pre-processed kwargs ready for an HTTP client call.

    Adapters call ``TransportKwargs.from_request(request)`` instead of
    manually handling ``extra_body`` / ``extra_headers`` on every request type.

    - ``body``: API-level keyword arguments with ``extra_body`` keys merged
      into the top level (mirroring how LiteLLM flattens them).
    - ``headers``: Extra HTTP headers to attach to the outgoing request.
    """

    _META_FIELDS: ClassVar[frozenset[str]] = frozenset({"extra_body", "extra_headers"})

    body: dict[str, Any]
    headers: dict[str, str]

    @classmethod
    def from_request(cls, request: Any, *, exclude: frozenset[str] = frozenset()) -> TransportKwargs:
        """Build transport-ready kwargs from a canonical request dataclass.

        1. Collects all non-None optional fields (respecting *exclude*).
        2. Pops ``extra_body`` and merges its keys into the top-level body dict.
        3. Pops ``extra_headers`` into a separate headers dict.
        """
        fields = cls._collect_optional_fields(request, exclude=exclude | cls._META_FIELDS)

        extra_body = getattr(request, "extra_body", None) or {}
        extra_headers = getattr(request, "extra_headers", None) or {}

        return cls(body={**fields, **extra_body}, headers=dict(extra_headers))

    @staticmethod
    def _collect_optional_fields(request: Any, *, exclude: frozenset[str] = frozenset()) -> dict[str, Any]:
        """Extract non-None optional fields from a request dataclass, skipping *exclude*.

        Targets fields whose default is ``None`` — i.e. truly optional kwargs
        the caller may or may not set.  Fields with non-``None`` defaults are
        not "optional" in this forwarding sense and are excluded.
        """
        return {
            f.name: v
            for f in fields(request)
            if f.name not in exclude and f.default is None and (v := getattr(request, f.name)) is not None
        }
