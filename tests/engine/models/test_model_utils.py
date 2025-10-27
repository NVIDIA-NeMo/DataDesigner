# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.engine.models.utils import prompt_to_messages, str_to_message


def test_str_to_message():
    assert str_to_message("hello") == {"content": "hello", "role": "user"}
    assert str_to_message("hello", role="system") == {"content": "hello", "role": "system"}
    assert str_to_message([{"type": "text", "text": "hello"}]) == {
        "content": [{"type": "text", "text": "hello"}],
        "role": "user",
    }
    assert str_to_message([{"type": "text", "text": "hello"}], role="system") == {
        "content": [{"type": "text", "text": "hello"}],
        "role": "system",
    }


def test_prompt_to_messages():
    stub_system_prompt = "some system prompt"
    mult_modal_context = {"type": "image_url", "image_url": {"url": "http://example.com/image.png"}}
    assert prompt_to_messages(user_prompt="hello") == [{"content": "hello", "role": "user"}]
    assert prompt_to_messages(user_prompt="hello", system_prompt=stub_system_prompt) == [
        {"content": stub_system_prompt, "role": "system"},
        {"content": "hello", "role": "user"},
    ]
    assert prompt_to_messages(user_prompt="hello", multi_modal_context=[mult_modal_context]) == [
        {"content": [{"type": "text", "text": "hello"}, mult_modal_context], "role": "user"}
    ]
    assert prompt_to_messages(
        user_prompt="hello", system_prompt=stub_system_prompt, multi_modal_context=[mult_modal_context]
    ) == [
        {"content": stub_system_prompt, "role": "system"},
        {"content": [{"type": "text", "text": "hello"}, mult_modal_context], "role": "user"},
    ]
