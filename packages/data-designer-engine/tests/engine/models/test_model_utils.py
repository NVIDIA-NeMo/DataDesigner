# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.config.models import Modality
from data_designer.config.utils.media_helpers import get_media_base64_context, get_media_url_context
from data_designer.engine.models.clients.types import AssistantMessage, ChatCompletionChoice, ChatCompletionResponse
from data_designer.engine.models.utils import (
    ChatMessage,
    GenerationTruncationReason,
    classify_generation_truncation_reason,
    merge_conversation_truncation_reason,
    prompt_to_messages,
)


def _make_response(
    *,
    finish_reason: str | None = None,
    raw: object | None = None,
) -> ChatCompletionResponse:
    message = AssistantMessage()
    return ChatCompletionResponse(
        message=message,
        raw=raw,
        choices=[ChatCompletionChoice(message=message, finish_reason=finish_reason)],
    )


def test_prompt_to_messages() -> None:
    stub_system_prompt = "some system prompt"
    mult_modal_context = {"type": "image_url", "image_url": {"url": "http://example.com/image.png"}}
    assert prompt_to_messages(user_prompt="hello") == [ChatMessage.as_user("hello")]
    assert prompt_to_messages(user_prompt="hello", system_prompt=stub_system_prompt) == [
        ChatMessage.as_system(stub_system_prompt),
        ChatMessage.as_user("hello"),
    ]
    assert prompt_to_messages(user_prompt="hello", multi_modal_context=[mult_modal_context]) == [
        ChatMessage.as_user([mult_modal_context, {"type": "text", "text": "hello"}])
    ]
    assert prompt_to_messages(
        user_prompt="hello", system_prompt=stub_system_prompt, multi_modal_context=[mult_modal_context]
    ) == [
        ChatMessage.as_system(stub_system_prompt),
        ChatMessage.as_user([mult_modal_context, {"type": "text", "text": "hello"}]),
    ]


def test_chat_message_as_tool_accepts_multimodal_content() -> None:
    content = [
        {"type": "text", "text": "Rendered chart:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
    ]

    message = ChatMessage.as_tool(content=content, tool_call_id="call-1")

    assert message.content == content
    assert message.to_dict()["content"] == content


def test_prompt_to_messages_preserves_mixed_media_context_order() -> None:
    context = [
        get_media_url_context(Modality.IMAGE.value, "https://example.com/image.png"),
        get_media_base64_context(Modality.AUDIO.value, "audio/mpeg", "abc123"),
        get_media_url_context(Modality.VIDEO.value, "https://example.com/video.mp4"),
    ]

    assert prompt_to_messages(user_prompt="describe", multi_modal_context=context) == [
        ChatMessage.as_user([*context, {"type": "text", "text": "describe"}])
    ]


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        pytest.param("length", GenerationTruncationReason.MAX_TOKENS, id="openai-max-tokens"),
        pytest.param("max_tokens", GenerationTruncationReason.MAX_TOKENS, id="anthropic-max-tokens"),
        pytest.param(
            "model_context_window_exceeded",
            GenerationTruncationReason.MODEL_CONTEXT_WINDOW_EXCEEDED,
            id="anthropic-context-window",
        ),
        pytest.param("stop", None, id="not-truncated"),
    ],
)
def test_classify_generation_truncation_reason_from_canonical_choice(
    finish_reason: str,
    expected: GenerationTruncationReason | None,
) -> None:
    response = _make_response(finish_reason=finish_reason)

    assert classify_generation_truncation_reason(response) is expected


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param({"choices": [{"finish_reason": "length"}]}, id="openai"),
        pytest.param({"stop_reason": "max_tokens"}, id="anthropic"),
    ],
)
def test_classify_generation_truncation_reason_uses_raw_fallback(raw: dict[str, object]) -> None:
    response = _make_response(raw=raw)

    assert classify_generation_truncation_reason(response) is GenerationTruncationReason.MAX_TOKENS


def test_classify_generation_truncation_reason_prefers_canonical_choice() -> None:
    response = _make_response(finish_reason="stop", raw={"stop_reason": "max_tokens"})

    assert classify_generation_truncation_reason(response) is None


@pytest.mark.parametrize(
    ("accumulated", "current", "expected"),
    [
        pytest.param(None, None, None, id="no-reason"),
        pytest.param(None, GenerationTruncationReason.MAX_TOKENS, GenerationTruncationReason.MAX_TOKENS),
        pytest.param(GenerationTruncationReason.MAX_TOKENS, None, GenerationTruncationReason.MAX_TOKENS),
        pytest.param(
            GenerationTruncationReason.MAX_TOKENS,
            GenerationTruncationReason.MODEL_CONTEXT_WINDOW_EXCEEDED,
            GenerationTruncationReason.MODEL_CONTEXT_WINDOW_EXCEEDED,
            id="current-context-window-wins",
        ),
        pytest.param(
            GenerationTruncationReason.MODEL_CONTEXT_WINDOW_EXCEEDED,
            GenerationTruncationReason.MAX_TOKENS,
            GenerationTruncationReason.MODEL_CONTEXT_WINDOW_EXCEEDED,
            id="accumulated-context-window-wins",
        ),
    ],
)
def test_merge_conversation_truncation_reason(
    accumulated: GenerationTruncationReason | None,
    current: GenerationTruncationReason | None,
    expected: GenerationTruncationReason | None,
) -> None:
    assert merge_conversation_truncation_reason(accumulated, current) is expected
