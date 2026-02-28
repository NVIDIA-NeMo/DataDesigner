# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any, Protocol

from data_designer.config.utils.image_helpers import (
    extract_base64_from_data_uri,
    is_base64_image,
    load_image_url_to_base64,
)
from data_designer.engine.models.clients.base import ModelClient
from data_designer.engine.models.clients.types import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImagePayload,
    ToolCall,
    Usage,
)

logger = logging.getLogger(__name__)


class LiteLLMRouter(Protocol):
    """Structural type for the LiteLLM router methods the bridge depends on."""

    def completion(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...

    async def acompletion(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...

    def embedding(self, *, model: str, input: list[str], **kwargs: Any) -> Any: ...

    async def aembedding(self, *, model: str, input: list[str], **kwargs: Any) -> Any: ...

    def image_generation(self, *, prompt: str, model: str, **kwargs: Any) -> Any: ...

    async def aimage_generation(self, *, prompt: str, model: str, **kwargs: Any) -> Any: ...


class LiteLLMBridgeClient(ModelClient):
    """Bridge adapter that wraps the existing LiteLLM router behind canonical client types."""

    # "messages" and "prompt" have None defaults but are passed explicitly to choose
    # between the chat-completion and diffusion code paths, so exclude them from the
    # automatic optional-field forwarding.
    _IMAGE_EXCLUDE = frozenset({"messages", "prompt"})

    def __init__(self, *, provider_name: str, router: LiteLLMRouter) -> None:
        self.provider_name = provider_name
        self._router = router

    def supports_chat_completion(self) -> bool:
        return True

    def supports_embeddings(self) -> bool:
        return True

    def supports_image_generation(self) -> bool:
        return True

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        response = self._router.completion(
            model=request.model,
            messages=request.messages,
            **_collect_non_none_optional_fields(request),
        )
        return _parse_chat_completion_response(response)

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        response = await self._router.acompletion(
            model=request.model,
            messages=request.messages,
            **_collect_non_none_optional_fields(request),
        )
        return _parse_chat_completion_response(response)

    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        response = self._router.embedding(
            model=request.model,
            input=request.inputs,
            **_collect_non_none_optional_fields(request),
        )
        vectors = [_extract_embedding_vector(item) for item in getattr(response, "data", [])]
        return EmbeddingResponse(vectors=vectors, usage=_extract_usage(getattr(response, "usage", None)), raw=response)

    async def aembeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        response = await self._router.aembedding(
            model=request.model,
            input=request.inputs,
            **_collect_non_none_optional_fields(request),
        )
        vectors = [_extract_embedding_vector(item) for item in getattr(response, "data", [])]
        return EmbeddingResponse(vectors=vectors, usage=_extract_usage(getattr(response, "usage", None)), raw=response)

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        image_kwargs = _collect_non_none_optional_fields(request, exclude=self._IMAGE_EXCLUDE)
        if request.messages is not None:
            response = self._router.completion(
                model=request.model,
                messages=request.messages,
                **image_kwargs,
            )
            images = _extract_images_from_chat_response(response)
        else:
            response = self._router.image_generation(
                prompt=request.prompt,
                model=request.model,
                **image_kwargs,
            )
            images = _extract_images_from_image_response(response)

        usage = _extract_usage(getattr(response, "usage", None), generated_images=len(images))
        return ImageGenerationResponse(images=images, usage=usage, raw=response)

    async def agenerate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        image_kwargs = _collect_non_none_optional_fields(request, exclude=self._IMAGE_EXCLUDE)
        if request.messages is not None:
            response = await self._router.acompletion(
                model=request.model,
                messages=request.messages,
                **image_kwargs,
            )
            images = _extract_images_from_chat_response(response)
        else:
            response = await self._router.aimage_generation(
                prompt=request.prompt,
                model=request.model,
                **image_kwargs,
            )
            images = _extract_images_from_image_response(response)

        usage = _extract_usage(getattr(response, "usage", None), generated_images=len(images))
        return ImageGenerationResponse(images=images, usage=usage, raw=response)

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


def _parse_chat_completion_response(response: Any) -> ChatCompletionResponse:
    first_choice = _first_or_none(getattr(response, "choices", None))
    message = _value_from(first_choice, "message")
    tool_calls = _extract_tool_calls(_value_from(message, "tool_calls"))
    images = _extract_images_from_chat_message(message)
    assistant_message = AssistantMessage(
        content=_coerce_message_content(_value_from(message, "content")),
        reasoning_content=_value_from(message, "reasoning_content"),
        tool_calls=tool_calls,
        images=images,
    )
    usage = _extract_usage(getattr(response, "usage", None), generated_images=len(images) if images else None)
    return ChatCompletionResponse(message=assistant_message, usage=usage, raw=response)


def _collect_non_none_optional_fields(request: Any, *, exclude: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Extract non-None optional fields from a request dataclass, skipping *exclude*."""
    return {
        f.name: v
        for f in dataclasses.fields(request)
        if f.name not in exclude and f.default is None and (v := getattr(request, f.name)) is not None
    }


def _extract_embedding_vector(item: Any) -> list[float]:
    value = _value_from(item, "embedding")
    if isinstance(value, list):
        return [float(v) for v in value]
    return []


def _extract_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
    if not raw_tool_calls:
        return []

    normalized_tool_calls: list[ToolCall] = []
    for raw_tool_call in raw_tool_calls:
        tool_call_id = _value_from(raw_tool_call, "id") or ""
        function = _value_from(raw_tool_call, "function")
        name = _value_from(function, "name") or ""
        arguments_value = _value_from(function, "arguments")
        arguments_json = _serialize_tool_arguments(arguments_value)
        normalized_tool_calls.append(ToolCall(id=str(tool_call_id), name=str(name), arguments_json=arguments_json))

    return normalized_tool_calls


def _serialize_tool_arguments(arguments_value: Any) -> str:
    if arguments_value is None:
        return "{}"
    if isinstance(arguments_value, str):
        return arguments_value
    try:
        return json.dumps(arguments_value)
    except Exception:
        return str(arguments_value)


def _extract_images_from_chat_response(response: Any) -> list[ImagePayload]:
    first_choice = _first_or_none(getattr(response, "choices", None))
    message = _value_from(first_choice, "message")
    return _extract_images_from_chat_message(message)


def _extract_images_from_chat_message(message: Any) -> list[ImagePayload]:
    images: list[ImagePayload] = []

    raw_images = _value_from(message, "images")
    if isinstance(raw_images, list):
        for raw_image in raw_images:
            parsed_image = _parse_image_payload(raw_image)
            if parsed_image is not None:
                images.append(parsed_image)

    if images:
        return images

    raw_content = _value_from(message, "content")
    if isinstance(raw_content, str):
        parsed_image = _parse_image_payload(raw_content)
        if parsed_image is not None:
            images.append(parsed_image)

    return images


def _extract_images_from_image_response(response: Any) -> list[ImagePayload]:
    images: list[ImagePayload] = []
    for raw_image in getattr(response, "data", []):
        parsed_image = _parse_image_payload(raw_image)
        if parsed_image is not None:
            images.append(parsed_image)
    return images


def _parse_image_payload(raw_image: Any) -> ImagePayload | None:
    try:
        if isinstance(raw_image, str):
            return _parse_image_string(raw_image)

        if isinstance(raw_image, dict):
            if "b64_json" in raw_image and isinstance(raw_image["b64_json"], str):
                return ImagePayload(b64_data=raw_image["b64_json"], mime_type=None)
            if "image_url" in raw_image:
                return _parse_image_payload(raw_image["image_url"])
            if "url" in raw_image and isinstance(raw_image["url"], str):
                return _parse_image_string(raw_image["url"])

        b64_json = _value_from(raw_image, "b64_json")
        if isinstance(b64_json, str):
            return ImagePayload(b64_data=b64_json, mime_type=None)

        url = _value_from(raw_image, "url")
        if isinstance(url, str):
            return _parse_image_string(url)
    except Exception:
        logger.debug("Unable to parse image payload from bridge response object.", exc_info=True)

    return None


def _parse_image_string(raw_value: str) -> ImagePayload | None:
    if raw_value.startswith("data:image/"):
        return ImagePayload(
            b64_data=extract_base64_from_data_uri(raw_value),
            mime_type=_extract_mime_type_from_data_uri(raw_value),
        )

    if is_base64_image(raw_value):
        return ImagePayload(b64_data=raw_value, mime_type=None)

    if raw_value.startswith(("http://", "https://")):
        b64_data = load_image_url_to_base64(raw_value)
        return ImagePayload(b64_data=b64_data, mime_type=None)

    return None


def _extract_mime_type_from_data_uri(data_uri: str) -> str | None:
    if not data_uri.startswith("data:"):
        return None
    head = data_uri.split(",", maxsplit=1)[0]
    if ";" in head:
        return head[5:].split(";", maxsplit=1)[0]
    return head[5:] or None


def _extract_usage(raw_usage: Any, generated_images: int | None = None) -> Usage | None:
    if raw_usage is None and generated_images is None:
        return None

    input_tokens = _value_from(raw_usage, "prompt_tokens")
    output_tokens = _value_from(raw_usage, "completion_tokens")
    total_tokens = _value_from(raw_usage, "total_tokens")

    if input_tokens is None:
        input_tokens = _value_from(raw_usage, "input_tokens")
    if output_tokens is None:
        output_tokens = _value_from(raw_usage, "output_tokens")

    if total_tokens is None and isinstance(input_tokens, int) and isinstance(output_tokens, int):
        total_tokens = input_tokens + output_tokens

    if generated_images is None:
        generated_images = _value_from(raw_usage, "generated_images")
    if generated_images is None and raw_usage is not None:
        generated_images = _value_from(raw_usage, "images")

    if input_tokens is None and output_tokens is None and total_tokens is None and generated_images is None:
        return None

    return Usage(
        input_tokens=_to_int_or_none(input_tokens),
        output_tokens=_to_int_or_none(output_tokens),
        total_tokens=_to_int_or_none(total_tokens),
        generated_images=_to_int_or_none(generated_images),
    )


def _coerce_message_content(content: Any) -> str | None:
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


def _to_int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _value_from(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _first_or_none(values: Any) -> Any | None:
    if isinstance(values, list) and values:
        return values[0]
    return None
