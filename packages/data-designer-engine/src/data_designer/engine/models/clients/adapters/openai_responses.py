# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from typing import Any

from data_designer.engine.models.clients.adapters.openai_compatible import (
    OpenAICompatibleClient,
    translate_openai_compatible_content_blocks,
)
from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind
from data_designer.engine.models.clients.parsing import extract_usage
from data_designer.engine.models.clients.types import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ProviderReplayData,
    ToolCall,
    TransportKwargs,
)


class OpenAIResponsesClient(OpenAICompatibleClient):
    """OpenAI-compatible client with explicit Responses routing."""

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if request.api == "chat-completions":
            return super().completion(request)
        payload = self._build_payload_or_raise(request)
        response = self._post_sync("/responses", payload, request.extra_headers or {}, request.model, request.timeout)
        return self._parse_response_or_raise(response, request.model)

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if request.api == "chat-completions":
            return await super().acompletion(request)
        payload = self._build_payload_or_raise(request)
        response = await self._apost("/responses", payload, request.extra_headers or {}, request.model, request.timeout)
        return self._parse_response_or_raise(response, request.model)

    def _build_payload_or_raise(self, request: ChatCompletionRequest) -> dict[str, Any]:
        try:
            return build_responses_payload(request)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ProviderError(
                kind=ProviderErrorKind.BAD_REQUEST,
                message=str(exc),
                provider_name=self.provider_name,
                model_name=request.model,
                cause=exc,
            ) from exc

    def _parse_response_or_raise(self, response: dict[str, Any], model: str) -> ChatCompletionResponse:
        try:
            return parse_responses_response(response)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ProviderError(
                kind=ProviderErrorKind.API_ERROR,
                message=f"Invalid Responses output: {exc}",
                provider_name=self.provider_name,
                model_name=model,
                cause=exc,
            ) from exc


def build_responses_payload(request: ChatCompletionRequest) -> dict[str, Any]:
    """Translate canonical requests to stateless Responses requests."""
    body = deepcopy(TransportKwargs.from_request(request).body)
    for key in ("stop", "seed", "frequency_penalty", "presence_penalty", "logprobs", "logit_bias"):
        if key in body:
            raise ValueError(f"Responses does not support {key!r}.")
    if body.pop("n", 1) != 1:
        raise ValueError("Responses supports only n=1.")
    for key in ("input", "messages", "model", "previous_response_id", "conversation", "instructions"):
        if key in body:
            raise ValueError(f"Responses does not support overriding {key!r}.")
    for key in ("stream", "background"):
        if body.pop(key, False) is not False:
            raise ValueError(f"Responses does not support {key}=true.")
    if body.setdefault("store", False) is not False:
        raise ValueError("Responses requires store=false; conversation state is replayed locally.")
    include = body.get("include", [])
    if not isinstance(include, list) or not all(isinstance(item, str) for item in include):
        raise ValueError("Responses include must be a list of strings.")
    body["include"] = list(dict.fromkeys([*include, "reasoning.encrypted_content"]))
    limits = [key for key in ("max_tokens", "max_completion_tokens", "max_output_tokens") if key in body]
    if len(limits) > 1:
        raise ValueError("Specify only one token limit for Responses.")
    if limits:
        body["max_output_tokens"] = body.pop(limits[0])
    if "reasoning_effort" in body:
        reasoning = dict(body.get("reasoning") or {})
        if "effort" in reasoning:
            raise ValueError("Specify reasoning.effort or reasoning_effort, not both.")
        reasoning["effort"] = body.pop("reasoning_effort")
        body["reasoning"] = reasoning
    if "response_format" in body:
        response_format = body.pop("response_format")
        text = dict(body.get("text") or {})
        if "format" in text:
            raise ValueError("Specify response_format or text.format, not both.")
        if response_format.get("type") == "json_schema":
            text["format"] = {"type": "json_schema", **response_format["json_schema"]}
        elif response_format.get("type") in {"text", "json_object"}:
            text["format"] = response_format
        else:
            raise ValueError("Unsupported Responses response_format.")
        body["text"] = text
    if "tools" in body:
        tools = []
        for tool in body["tools"]:
            if tool.get("type") != "function":
                raise ValueError("Responses adapter supports only function tools.")
            function = tool["function"]
            tools.append({"type": "function", "strict": False, **function})
        body["tools"] = tools
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        if choice.get("type") != "function":
            raise ValueError("Responses adapter supports only function tool choices.")
        body["tool_choice"] = {"type": "function", "name": choice["function"]["name"]}
    return {"model": request.model, "input": translate_responses_messages(request), **body}


def translate_responses_messages(request: ChatCompletionRequest) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, message in enumerate(request.messages):
        role = message.get("role")
        replay = request.message_replay_data[index] if request.message_replay_data else None
        if replay is not None:
            if role != "assistant" or replay.provider_type != "openai-responses":
                raise ValueError("Responses replay data requires an assistant turn from openai-responses.")
            if not isinstance(replay.payload, list) or not all(isinstance(item, dict) for item in replay.payload):
                raise ValueError("Responses replay payload must be a list of output items.")
            items.extend(deepcopy(replay.payload))
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not call_id:
                raise ValueError("Responses tool results require tool_call_id.")
            content = message.get("content")
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": content if isinstance(content, str) else translate_responses_content(content),
                }
            )
        elif role in {"user", "system", "developer", "assistant"}:
            content = message.get("content")
            if content is not None:
                items.append(
                    {"role": role, "content": translate_responses_content(content, assistant=role == "assistant")}
                )
            for call in message.get("tool_calls") or []:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    }
                )
        else:
            raise ValueError(f"Unsupported Responses message role: {role!r}.")
    return items


def translate_responses_content(content: Any, *, assistant: bool = False) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise ValueError("Responses content must be text or a list of content blocks.")
    translated = []
    for block in translate_openai_compatible_content_blocks(content):
        if not isinstance(block, dict):
            raise ValueError("Responses content blocks must be objects.")
        if block.get("type") == "text":
            translated.append({"type": "input_text", "text": block["text"]})
        elif block.get("type") == "image_url" and not assistant:
            image = block["image_url"]
            translated.append({"type": "input_image", "image_url": image["url"], "detail": image.get("detail", "auto")})
        else:
            raise ValueError(f"Unsupported Responses content block: {block.get('type')!r}.")
    return "\n".join(block["text"] for block in translated) if assistant else translated


def parse_responses_response(response: dict[str, Any]) -> ChatCompletionResponse:
    if not isinstance(response, dict):
        raise ValueError("Responses body must be an object.")
    if response.get("error") or response.get("status") not in {"completed", "incomplete"}:
        raise ValueError(f"Responses request failed: {response.get('error') or response.get('status')}")
    output = response.get("output") or []
    if not isinstance(output, list) or not all(isinstance(item, dict) for item in output):
        raise ValueError("Responses output must be a list of items.")
    texts: list[str] = []
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    refused = False
    for item in output:
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    texts.append(block["text"])
                elif block.get("type") == "refusal":
                    refused = True
        elif item.get("type") == "reasoning":
            reasoning.extend(block["text"] for block in item.get("summary", []) if block.get("type") == "summary_text")
        elif item.get("type") == "function_call":
            if response.get("status") == "incomplete" or item.get("status", "completed") != "completed":
                raise ValueError(
                    "Incomplete Responses function call; no tools were executed. Check the output token limit."
                )
            calls.append(ToolCall(id=item["call_id"], name=item["name"], arguments_json=item["arguments"]))
    message = AssistantMessage(
        content="".join(texts) if texts else None,
        reasoning_content="\n".join(reasoning) if reasoning else None,
        tool_calls=calls,
        replay_data=ProviderReplayData(provider_type="openai-responses", payload=deepcopy(output)),
    )
    finish_reason = "tool_calls" if calls else "stop"
    if response.get("status") == "incomplete":
        reason = (response.get("incomplete_details") or {}).get("reason")
        finish_reason = "length" if reason == "max_output_tokens" else reason
    elif refused:
        finish_reason = "content_filter"
    return ChatCompletionResponse(
        message=message,
        usage=extract_usage(response.get("usage")),
        raw=response,
        choices=[ChatCompletionChoice(message=message, index=0, finish_reason=finish_reason)],
    )
