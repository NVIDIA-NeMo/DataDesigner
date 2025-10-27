# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


def prompt_to_messages(
    *,
    user_prompt: str,
    system_prompt: str | None = None,
    multi_modal_context: list[dict[str, Any]] | None = None,
) -> list[dict[str, str | list[dict]]]:
    """Convert a user and system prompt into Messages format.

    Args:
        user_prompt (str): A user prompt.
        system_prompt (str, optional): An optional system prompt.
    """
    user_content = user_prompt
    if multi_modal_context and len(multi_modal_context) > 0:
        user_content = []
        user_content.append({"type": "text", "text": user_prompt})
        for context in multi_modal_context:
            user_content.append(context)
    return (
        [
            str_to_message(content=system_prompt, role="system"),
            str_to_message(content=user_content, role="user"),
        ]
        if system_prompt
        else [str_to_message(content=user_content, role="user")]
    )


def str_to_message(content: str | list[dict], role: str = "user") -> dict[str, str | list[dict]]:
    return {"content": content, "role": role}
