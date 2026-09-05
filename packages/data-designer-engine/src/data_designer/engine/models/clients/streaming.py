# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble SSE chat responses before exposing them to parsers or tool execution."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, NoReturn

from data_designer.engine.models.clients.errors import (
    ProviderError,
    ProviderErrorKind,
    map_http_status_to_provider_error_kind,
)


class ChatStream(ABC):
    """Decode SSE frames and enforce an explicit provider end-of-stream marker."""

    def __init__(self, provider_name: str, model_name: str) -> None:
        """Initialize an empty response and error context for the named provider/model."""
        self.provider_name: str = provider_name
        self.model_name: str = model_name
        self.complete: bool = False
        self._data: list[str] = []
        self._event: str = ""

    def feed_line(self, line: str) -> bool:
        """Consume one decoded SSE line; return whether the response has ended.

        Args:
            line: UTF-8 line without its newline, including blank frame separators.

        Raises:
            ProviderError: For malformed payloads or provider errors inside a stream.
        """
        line = line.removeprefix("\ufeff")
        if line:
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "data":
                self._data.append(value)
            elif field == "event":
                self._event = value
            return self.complete

        data, event = "\n".join(self._data), self._event
        self._data, self._event = [], ""
        if not data:
            return self.complete
        try:
            payload = {"type": "done"} if data == "[DONE]" else json.loads(data)
            if not isinstance(payload, dict):
                self.fail("Expected a JSON object in the stream")
            if event == "error" or payload.get("error") or payload.get("type") == "error":
                self._raise_provider_error(payload)
            self._consume(payload)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            self.fail("Malformed chat stream event", cause=exc)
        return self.complete

    def finish(self) -> dict[str, Any]:
        """Return the assembled response, or fail if the connection ended prematurely.

        Incomplete responses are connection failures so the request executor can retry
        the entire request with a new accumulator; no partial result is returned.
        """
        if not self.complete:
            self.fail("Chat stream ended before its completion marker", kind=ProviderErrorKind.API_CONNECTION)
        return self._build_response()

    def fail(
        self,
        message: str,
        *,
        kind: ProviderErrorKind = ProviderErrorKind.API_ERROR,
        cause: Exception | None = None,
    ) -> NoReturn:
        """Raise a canonical error using message, failure kind, and optional original cause."""
        raise ProviderError(kind, message, provider_name=self.provider_name, model_name=self.model_name, cause=cause)

    def _raise_provider_error(self, payload: dict[str, Any]) -> NoReturn:
        """Raise the error carried by an SSE payload, preserving rate-limit classification."""
        error = payload.get("error") or payload
        if not isinstance(error, dict):
            self.fail(str(error))
        status = {
            "rate_limit_error": 429,
            "overloaded_error": 529,
            "authentication_error": 401,
            "permission_error": 403,
            "invalid_request_error": 400,
            "api_error": 500,
        }.get(error.get("type"))
        code = error.get("code")
        if status is None and str(code).isdigit():
            status = int(code)
        message = str(error.get("message") or "Provider reported an error in the chat stream")
        kind = map_http_status_to_provider_error_kind(status, message) if status else ProviderErrorKind.API_ERROR
        raise ProviderError(
            kind, message, status_code=status, provider_name=self.provider_name, model_name=self.model_name
        )

    @abstractmethod
    def _consume(self, payload: dict[str, Any]) -> None:
        """Merge one provider event into this response, setting complete at its terminal event."""

    @abstractmethod
    def _build_response(self) -> dict[str, Any]:
        """Validate accumulated state and return a non-streaming provider response object."""


class OpenAIChatStream(ChatStream):
    """Collect interleaved choices, reasoning, tool arguments, and final token usage."""

    def __init__(self, provider_name: str, model_name: str, num_choices: int = 1) -> None:
        """Initialize named provider/model context and the expected number of choices."""
        super().__init__(provider_name, model_name)
        self._num_choices: int = num_choices
        self._response: dict[str, Any] = {}
        self._choices: dict[int, dict[str, Any]] = {}

    def _consume(self, payload: dict[str, Any]) -> None:
        """Merge one chat chunk into indexed choices, or record the [DONE] marker."""
        if payload.get("type") == "done":
            self.complete = True
            return
        for key, value in payload.items():
            if key != "choices" and value is not None:
                self._response[key] = value
        for chunk in payload.get("choices") or []:
            index = chunk["index"]
            if type(index) is not int or index < 0:
                self.fail("Invalid choice index in chat stream")
            choice = self._choices.setdefault(index, {"index": index, "message": {}})
            merge_stream_delta(choice["message"], chunk.get("delta") or {})
            if chunk.get("finish_reason") is not None:
                choice["finish_reason"] = chunk["finish_reason"]
            if chunk.get("logprobs") is not None:
                merge_stream_delta(choice.setdefault("logprobs", {}), chunk["logprobs"])

    def _build_response(self) -> dict[str, Any]:
        """Return all choices in index order, rejecting missing or unfinished choices."""
        if set(self._choices) != set(range(self._num_choices)) or any(
            not choice.get("finish_reason") for choice in self._choices.values()
        ):
            self.fail("Chat stream ended with missing or unfinished choices", kind=ProviderErrorKind.API_CONNECTION)
        for choice in self._choices.values():
            calls = choice["message"].get("tool_calls")
            if calls:
                calls.sort(key=lambda call: call["index"])
        return {
            **self._response,
            "object": "chat.completion",
            "choices": [self._choices[i] for i in sorted(self._choices)],
        }


def merge_stream_delta(target: dict[str, Any], delta: dict[str, Any]) -> None:
    """Merge a JSON delta into target in place, concatenating text and indexed list items.

    Args:
        target: Accumulated message or nested object, mutated by this function.
        delta: New fragments; null fields are ignored. Lists with index fields are
            merged by index, allowing tool calls and content blocks to interleave.

    Raises:
        TypeError: If a provider changes the type of a field between chunks.
    """
    for key, value in delta.items():
        if value is None:
            continue
        if key not in target or target[key] is None:
            target[key] = deepcopy(value)
        elif isinstance(value, str):
            target[key] += value
        elif isinstance(value, dict):
            merge_stream_delta(target[key], value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "index" in item:
                    prior = next(
                        (old for old in target[key] if isinstance(old, dict) and old.get("index") == item["index"]),
                        None,
                    )
                    if prior is not None:
                        merge_stream_delta(prior, item)
                        continue
                target[key].append(deepcopy(item))
        else:
            target[key] = value


class AnthropicMessageStream(ChatStream):
    """Collect Messages API content blocks and cumulative usage through message_stop."""

    def __init__(self, provider_name: str, model_name: str) -> None:
        """Initialize named provider/model context and empty message/block state."""
        super().__init__(provider_name, model_name)
        self._message: dict[str, Any] | None = None
        self._blocks: dict[int, dict[str, Any]] = {}
        self._partial_json: dict[int, list[str]] = {}
        self._open_blocks: set[int] = set()

    def _consume(self, payload: dict[str, Any]) -> None:
        """Apply one Messages event, preserving thinking signatures and tool JSON."""
        event = payload.get("type")
        if event == "message_start":
            if self._message is not None:
                self.fail("Duplicate message_start in chat stream")
            self._message = deepcopy(payload["message"])
        elif event == "content_block_start":
            index = payload["index"]
            if type(index) is not int or index < 0 or index in self._blocks:
                self.fail("Invalid or duplicate content block index in chat stream")
            self._blocks[index] = deepcopy(payload["content_block"])
            self._open_blocks.add(index)
        elif event == "content_block_delta":
            index, delta = payload["index"], payload["delta"]
            if index not in self._open_blocks:
                self.fail("Delta for an unopened content block")
            if delta.get("type") == "input_json_delta":
                self._partial_json.setdefault(index, []).append(delta["partial_json"])
            elif delta.get("type") == "citations_delta":
                self._blocks[index].setdefault("citations", []).append(delta["citation"])
            else:
                merge_stream_delta(self._blocks[index], {key: value for key, value in delta.items() if key != "type"})
        elif event == "content_block_stop":
            index = payload["index"]
            self._open_blocks.remove(index)
            if index in self._partial_json:
                self._blocks[index]["input"] = json.loads("".join(self._partial_json.pop(index)))
        elif event == "message_delta":
            if self._message is None:
                self.fail("message_delta arrived before message_start")
            self._message.update(payload.get("delta") or {})
            self._message.setdefault("usage", {}).update(
                {key: value for key, value in (payload.get("usage") or {}).items() if value is not None}
            )
        elif event == "message_stop":
            self.complete = True
        # Ping and future event types do not change the accumulated message.

    def _build_response(self) -> dict[str, Any]:
        """Return the final message with ordered blocks and non-duplicated token counts."""
        if (
            self._message is None
            or self._open_blocks
            or not self._message.get("stop_reason")
            or set(self._blocks) != set(range(len(self._blocks)))
        ):
            self.fail("Chat stream ended with an unfinished message", kind=ProviderErrorKind.API_CONNECTION)
        return {**self._message, "content": [self._blocks[i] for i in sorted(self._blocks)]}
