# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from typing import Any

from data_designer.engine.models.clients.parsing import extract_usage
from data_designer.engine.models.clients.types import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ToolCall,
    Usage,
)

_DEFAULT_MAX_TOKENS = 4096
_DATA_URI_RE = re.compile(r"^data:(?P<media_type>[^;]+);base64,(?P<data>.+)$")


def merge_system_parts(parts: list[str | list[dict[str, Any]]]) -> str | list[dict[str, Any]]:
    """Merge system parts into a single string or Anthropic block list.

    If every part is a plain string, join them with double newlines.
    Otherwise, normalize all parts into a flat list of content blocks.
    """
    if all(isinstance(p, str) for p in parts):
        return "\n\n".join(parts)  # type: ignore[arg-type]

    blocks: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, str):
            blocks.append({"type": "text", "text": part})
        else:
            blocks.extend(part)
    return blocks


def build_anthropic_payload(request: ChatCompletionRequest) -> dict[str, Any]:
    """Build an Anthropic Messages API payload from a canonical request."""
    system_parts, messages = translate_request_messages(request.messages)

    payload: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_tokens if request.max_tokens is not None else _DEFAULT_MAX_TOKENS,
    }

    if system_parts:
        payload["system"] = merge_system_parts(system_parts)

    if request.tools:
        payload["tools"] = [translate_tool_definition(tool) for tool in request.tools]

    if request.stop is not None:
        if isinstance(request.stop, str):
            payload["stop_sequences"] = [request.stop]
        else:
            payload["stop_sequences"] = list(request.stop)

    return payload


def parse_anthropic_response(response_json: dict[str, Any]) -> ChatCompletionResponse:
    """Convert an Anthropic Messages API response into canonical response types."""
    content_blocks = response_json.get("content") or []

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in content_blocks:
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if text:
                text_parts.append(text)
        elif block_type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments_json=json.dumps(block.get("input", {})),
                )
            )
        elif block_type == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                thinking_parts.append(thinking)

    message = AssistantMessage(
        content="\n".join(text_parts) if text_parts else None,
        reasoning_content="\n".join(thinking_parts) if thinking_parts else None,
        tool_calls=tool_calls,
    )

    raw_usage = response_json.get("usage")
    usage: Usage | None = None
    if raw_usage:
        usage = extract_usage(raw_usage)

    return ChatCompletionResponse(message=message, usage=usage, raw=response_json)


def translate_request_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[str | list[dict[str, Any]]], list[dict[str, Any]]]:
    system_parts: list[str | list[dict[str, Any]]] = []
    translated_messages: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            system_content = extract_system_content(msg.get("content"))
            if system_content is not None:
                system_parts.append(system_content)
            continue

        if role == "tool":
            pending_tool_results.append(translate_tool_result_message(msg))
            continue

        if pending_tool_results:
            translated_messages.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

        translated_messages.append(translate_non_tool_message(msg))

    if pending_tool_results:
        translated_messages.append({"role": "user", "content": pending_tool_results})

    return system_parts, translated_messages


def extract_system_content(content: Any) -> str | list[dict[str, Any]] | None:
    """Extract system content, preserving image blocks for multimodal system prompts.

    Returns a plain string when only text is present, or a list of Anthropic
    content blocks when non-text blocks (e.g. images) are included.
    """
    if isinstance(content, str):
        return content or None

    translated_blocks = translate_content_blocks(content)
    if not translated_blocks:
        return None

    has_non_text = any(not (isinstance(b, dict) and b.get("type") == "text") for b in translated_blocks)
    if has_non_text:
        return translated_blocks

    text_parts = [
        block.get("text", "")
        for block in translated_blocks
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
    ]
    if not text_parts:
        return None
    return "\n".join(text_parts)


def translate_non_tool_message(msg: dict[str, Any]) -> dict[str, Any]:
    role = msg.get("role")
    if role not in {"user", "assistant"}:
        raise ValueError(f"Anthropic adapter does not support message role {role!r}.")

    content = msg.get("content")
    if role == "assistant" and msg.get("tool_calls"):
        translated_content = translate_content_blocks(content)
        translated_content.extend(translate_tool_calls(msg.get("tool_calls")))
        return {"role": "assistant", "content": translated_content}

    return {"role": role, "content": translate_regular_content(content)}


def translate_regular_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    return translate_content_blocks(content)


def translate_content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        raw_blocks = content
    elif content is None:
        return []
    else:
        raw_blocks = [{"type": "text", "text": str(content)}]

    translated: list[dict[str, Any]] = []
    for block in raw_blocks:
        if isinstance(block, dict) and block.get("type") == "image_url":
            anthropic_block = translate_image_url_block(block)
            if anthropic_block is not None:
                translated.append(anthropic_block)
            continue
        translated.append(block)
    return translated


def translate_tool_definition(tool: dict[str, Any]) -> dict[str, Any]:
    if "name" in tool and "input_schema" in tool:
        translated_tool = {
            "name": tool["name"],
            "input_schema": tool.get("input_schema") or {"type": "object", "properties": {}},
        }
        description = tool.get("description")
        if isinstance(description, str) and description:
            translated_tool["description"] = description
        return translated_tool

    function = tool.get("function")
    if not isinstance(function, dict):
        raise ValueError(f"Anthropic tool definition must include a function payload, got: {tool!r}")

    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Anthropic tool definition is missing a function name, got: {tool!r}")

    parameters = function.get("parameters")
    if parameters is None:
        parameters = {"type": "object", "properties": {}}
    if not isinstance(parameters, dict):
        raise ValueError(f"Anthropic tool definition parameters must be a JSON schema object, got: {tool!r}")

    translated_tool = {
        "name": name,
        "input_schema": parameters,
    }
    description = function.get("description")
    if isinstance(description, str) and description:
        translated_tool["description"] = description
    return translated_tool


def translate_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        raise ValueError(f"Anthropic assistant tool calls must be a list, got: {tool_calls!r}")

    return [translate_tool_call(tool_call) for tool_call in tool_calls]


def translate_tool_call(tool_call: Any) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        raise ValueError(f"Anthropic assistant tool call must be an object, got: {tool_call!r}")

    if tool_call.get("type") == "tool_use":
        return tool_call

    tool_call_id = tool_call.get("id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise ValueError(f"Anthropic assistant tool call is missing an id, got: {tool_call!r}")

    if "function" in tool_call:
        function = tool_call.get("function")
        if not isinstance(function, dict):
            raise ValueError(f"Anthropic assistant tool call must include a function payload, got: {tool_call!r}")
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = tool_call.get("name")
        arguments = tool_call.get("arguments_json")

    if not isinstance(name, str) or not name:
        raise ValueError(f"Anthropic assistant tool call is missing a function name, got: {tool_call!r}")

    return {
        "type": "tool_use",
        "id": tool_call_id,
        "name": name,
        "input": parse_tool_call_arguments(arguments),
    }


def parse_tool_call_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        raise ValueError(f"Anthropic tool call arguments must be JSON or an object, got: {arguments!r}")

    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Anthropic tool call arguments must be valid JSON, got: {arguments!r}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Anthropic tool call arguments must decode to a JSON object, got: {arguments!r}")
    return parsed


def translate_tool_result_message(msg: dict[str, Any]) -> dict[str, Any]:
    tool_use_id = msg.get("tool_call_id")
    if not isinstance(tool_use_id, str) or not tool_use_id:
        raise ValueError(f"Anthropic tool result message is missing a tool_call_id, got: {msg!r}")

    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": translate_tool_result_content(msg.get("content")),
    }


def translate_tool_result_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content

    translated_blocks = translate_content_blocks(content)
    if all(isinstance(block, dict) and block.get("type") == "text" for block in translated_blocks):
        return "\n".join(block.get("text", "") for block in translated_blocks)
    return translated_blocks


def translate_image_url_block(block: dict[str, Any]) -> dict[str, Any] | None:
    image_url = block.get("image_url")
    if image_url is None:
        return None

    if isinstance(image_url, dict):
        url = image_url.get("url", "")
    else:
        url = str(image_url)

    match = _DATA_URI_RE.match(url)
    if match:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": match.group("media_type"),
                "data": match.group("data"),
            },
        }

    return {
        "type": "image",
        "source": {"type": "url", "url": url},
    }
