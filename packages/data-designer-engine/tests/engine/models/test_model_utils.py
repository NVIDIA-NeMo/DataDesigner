# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
