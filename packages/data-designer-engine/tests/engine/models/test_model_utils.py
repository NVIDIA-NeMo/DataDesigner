# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.models import Modality
from data_designer.config.utils.media_helpers import get_media_base64_context, get_media_url_context
from data_designer.engine.models.utils import ChatMessage, prompt_to_messages


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
