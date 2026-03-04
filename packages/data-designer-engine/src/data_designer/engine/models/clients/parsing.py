# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared response-parsing helpers reusable across provider adapters."""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any

from data_designer.config.utils.image_helpers import (
    aload_image_url_to_base64,
    extract_base64_from_data_uri,
    is_base64_image,
    load_image_url_to_base64,
)
from data_designer.engine.models.clients.types import (
    AssistantMessage,
    ChatCompletionResponse,
    ImagePayload,
    ToolCall,
    Usage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# High-level response parsers
# ---------------------------------------------------------------------------


def parse_chat_completion_response(response: Any) -> ChatCompletionResponse:
    first_choice = get_first_value_or_none(getattr(response, "choices", None))
    message = get_value_from(first_choice, "message")
    tool_calls = extract_tool_calls(get_value_from(message, "tool_calls"))
    images = extract_images_from_chat_message(message)
    assistant_message = AssistantMessage(
        content=coerce_message_content(get_value_from(message, "content")),
        reasoning_content=get_value_from(message, "reasoning_content"),
        tool_calls=tool_calls,
        images=images,
    )
    usage = extract_usage(getattr(response, "usage", None), generated_images=len(images) if images else None)
    return ChatCompletionResponse(message=assistant_message, usage=usage, raw=response)


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------


def extract_images_from_chat_response(response: Any) -> list[ImagePayload]:
    first_choice = get_first_value_or_none(getattr(response, "choices", None))
    message = get_value_from(first_choice, "message")
    return extract_images_from_chat_message(message)


async def aextract_images_from_chat_response(response: Any) -> list[ImagePayload]:
    first_choice = get_first_value_or_none(getattr(response, "choices", None))
    message = get_value_from(first_choice, "message")
    return await aextract_images_from_chat_message(message)


def extract_images_from_chat_message(message: Any) -> list[ImagePayload]:
    primary, fallback = collect_raw_image_candidates(message)
    images = parse_image_list(primary)
    return images if images else parse_image_list(fallback)


async def aextract_images_from_chat_message(message: Any) -> list[ImagePayload]:
    primary, fallback = collect_raw_image_candidates(message)
    images = await aparse_image_list(primary)
    return images if images else await aparse_image_list(fallback)


def extract_images_from_image_response(response: Any) -> list[ImagePayload]:
    return parse_image_list(getattr(response, "data", []))


async def aextract_images_from_image_response(response: Any) -> list[ImagePayload]:
    return await aparse_image_list(getattr(response, "data", []))


def collect_raw_image_candidates(message: Any) -> tuple[list[Any], list[Any]]:
    """Return (primary, fallback) raw image candidates from a message.

    Only string content is used as a fallback source.  List-format content blocks
    (e.g. OpenAI multimodal ``image_url`` items) are not extracted here; that
    parsing is deferred to adapter-specific logic in future PRs.
    """
    primary: list[Any] = []
    raw_images = get_value_from(message, "images")
    if isinstance(raw_images, list):
        primary = list(raw_images)

    fallback: list[Any] = []
    raw_content = get_value_from(message, "content")
    if isinstance(raw_content, str):
        fallback = [raw_content]

    return primary, fallback


def parse_image_list(raw_items: list[Any]) -> list[ImagePayload]:
    return [img for raw in raw_items if (img := parse_image_payload(raw)) is not None]


async def aparse_image_list(raw_items: list[Any]) -> list[ImagePayload]:
    return [img for raw in raw_items if (img := await aparse_image_payload(raw)) is not None]


# ---------------------------------------------------------------------------
# Image payload parsing
# ---------------------------------------------------------------------------


def parse_image_payload(raw_image: Any) -> ImagePayload | None:
    try:
        result = resolve_image_payload(raw_image)
        if isinstance(result, str):
            return ImagePayload(b64_data=load_image_url_to_base64(result), mime_type=None)
        return result
    except Exception:
        logger.debug("Unable to parse image payload from response object.", exc_info=True)
        return None


async def aparse_image_payload(raw_image: Any) -> ImagePayload | None:
    try:
        result = resolve_image_payload(raw_image)
        if isinstance(result, str):
            return ImagePayload(b64_data=await aload_image_url_to_base64(result), mime_type=None)
        return result
    except Exception:
        logger.debug("Unable to parse image payload from response object.", exc_info=True)
        return None


def resolve_image_payload(raw_image: Any) -> ImagePayload | str | None:
    """Resolve a raw image to an ImagePayload, a URL needing I/O, or None."""
    if isinstance(raw_image, str):
        return resolve_image_string(raw_image)

    if isinstance(raw_image, dict):
        if "b64_json" in raw_image and isinstance(raw_image["b64_json"], str):
            return ImagePayload(b64_data=raw_image["b64_json"], mime_type=None)
        if "image_url" in raw_image:
            return resolve_image_payload(raw_image["image_url"])
        if "url" in raw_image and isinstance(raw_image["url"], str):
            return resolve_image_string(raw_image["url"])

    b64_json = get_value_from(raw_image, "b64_json")
    if isinstance(b64_json, str):
        return ImagePayload(b64_data=b64_json, mime_type=None)

    url = get_value_from(raw_image, "url")
    if isinstance(url, str):
        return resolve_image_string(url)

    return None


def resolve_image_string(raw_value: str) -> ImagePayload | str | None:
    """Return an ImagePayload for inline data, a URL string for HTTP URLs, or None."""
    if raw_value.startswith("data:image/"):
        return ImagePayload(
            b64_data=extract_base64_from_data_uri(raw_value),
            mime_type=extract_mime_type_from_data_uri(raw_value),
        )

    if is_base64_image(raw_value):
        return ImagePayload(b64_data=raw_value, mime_type=None)

    if raw_value.startswith(("http://", "https://")):
        return raw_value

    return None


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------


def extract_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
    if not raw_tool_calls:
        return []

    normalized_tool_calls: list[ToolCall] = []
    for raw_tool_call in raw_tool_calls:
        tool_call_id = get_value_from(raw_tool_call, "id") or ""
        function = get_value_from(raw_tool_call, "function")
        name = get_value_from(function, "name") or ""
        arguments_value = get_value_from(function, "arguments")
        arguments_json = serialize_tool_arguments(arguments_value)
        normalized_tool_calls.append(ToolCall(id=str(tool_call_id), name=str(name), arguments_json=arguments_json))

    return normalized_tool_calls


def serialize_tool_arguments(arguments_value: Any) -> str:
    if arguments_value is None:
        return "{}"
    if isinstance(arguments_value, str):
        return arguments_value
    try:
        return json.dumps(arguments_value)
    except Exception:
        return str(arguments_value)


# ---------------------------------------------------------------------------
# Usage & content helpers
# ---------------------------------------------------------------------------


def extract_usage(raw_usage: Any, generated_images: int | None = None) -> Usage | None:
    if raw_usage is None and generated_images is None:
        return None

    input_tokens = get_value_from(raw_usage, "prompt_tokens")
    output_tokens = get_value_from(raw_usage, "completion_tokens")
    total_tokens = get_value_from(raw_usage, "total_tokens")

    if input_tokens is None:
        input_tokens = get_value_from(raw_usage, "input_tokens")
    if output_tokens is None:
        output_tokens = get_value_from(raw_usage, "output_tokens")

    if total_tokens is None and isinstance(input_tokens, int) and isinstance(output_tokens, int):
        total_tokens = input_tokens + output_tokens

    if generated_images is None:
        generated_images = get_value_from(raw_usage, "generated_images")
    if generated_images is None and raw_usage is not None:
        generated_images = get_value_from(raw_usage, "images")

    if input_tokens is None and output_tokens is None and total_tokens is None and generated_images is None:
        return None

    return Usage(
        input_tokens=coerce_to_int_or_none(input_tokens),
        output_tokens=coerce_to_int_or_none(output_tokens),
        total_tokens=coerce_to_int_or_none(total_tokens),
        generated_images=coerce_to_int_or_none(generated_images),
    )


def extract_embedding_vector(item: Any) -> list[float]:
    value = get_value_from(item, "embedding")
    if isinstance(value, list):
        return [float(v) for v in value]
    return []


def extract_mime_type_from_data_uri(data_uri: str) -> str | None:
    if not data_uri.startswith("data:"):
        return None
    head = data_uri.split(",", maxsplit=1)[0]
    if ";" in head:
        return head[5:].split(";", maxsplit=1)[0]
    return head[5:] or None


def coerce_message_content(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text_value = block.get("text")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
        if text_parts:
            return "\n".join(text_parts)
    return str(content)


def coerce_to_int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except (ValueError, OverflowError):
            return None
    return None


# ---------------------------------------------------------------------------
# Generic accessors
# ---------------------------------------------------------------------------


def get_value_from(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def get_first_value_or_none(values: Any) -> Any | None:
    if isinstance(values, list) and values:
        return values[0]
    return None


def collect_non_none_optional_fields(request: Any, *, exclude: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Extract non-None optional fields from a request dataclass, skipping *exclude*.

    The ``f.default is None`` check intentionally targets fields whose default is
    ``None`` — i.e. truly optional kwargs the caller may or may not set.  Fields with
    non-``None`` defaults are not "optional" in this forwarding sense and are excluded.
    """
    return {
        f.name: v
        for f in dataclasses.fields(request)
        if f.name not in exclude and f.default is None and (v := getattr(request, f.name)) is not None
    }
