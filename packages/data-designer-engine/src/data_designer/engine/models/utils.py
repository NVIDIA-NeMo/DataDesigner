# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ChatMessage:
    """A chat message in an LLM conversation.

    This dataclass represents messages exchanged in a conversation with an LLM,
    supporting various message types including user prompts, assistant responses,
    system instructions, and tool interactions.

    Attributes:
        role: The role of the message sender. One of 'user', 'assistant', 'system', or 'tool'.
        content: The message content. Can be a string or a list of content blocks
            for multimodal messages (e.g., text + images).
        reasoning_content: Optional reasoning/thinking content from the assistant,
            typically from extended thinking or chain-of-thought models.
        tool_calls: Optional list of tool calls requested by the assistant.
            Each tool call contains 'id', 'type', and 'function' keys.
        tool_call_id: Optional ID linking a tool response to its corresponding
            tool call. Required for messages with role='tool'.
    """

    role: Literal["user", "assistant", "system", "tool"]
    content: str | list[dict[str, Any]] = ""
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the message to a dictionary format for API calls.

        Returns:
            A dictionary containing the message fields. Only includes non-empty
            optional fields to keep the output clean.
        """
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.reasoning_content:
            result["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result

    @classmethod
    def as_user(cls, content: str | list[dict[str, Any]]) -> ChatMessage:
        """Create a user message."""
        return cls(role="user", content=content)

    @classmethod
    def as_assistant(
        cls,
        content: str = "",
        reasoning_content: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> ChatMessage:
        """Create an assistant message."""
        return cls(
            role="assistant",
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls or [],
        )

    @classmethod
    def as_system(cls, content: str) -> ChatMessage:
        """Create a system message."""
        return cls(role="system", content=content)

    @classmethod
    def as_tool(cls, content: str, tool_call_id: str) -> ChatMessage:
        """Create a tool response message."""
        return cls(role="tool", content=content, tool_call_id=tool_call_id)


def prompt_to_messages(
    *,
    user_prompt: str,
    system_prompt: str | None = None,
    multi_modal_context: list[dict[str, Any]] | None = None,
) -> list[ChatMessage]:
    """Convert a user and system prompt into ChatMessage list.

    Args:
        user_prompt (str): A user prompt.
        system_prompt (str, optional): An optional system prompt.
    """
    user_content: str | list[dict[str, Any]] = user_prompt
    if multi_modal_context:
        user_content = [*multi_modal_context, {"type": "text", "text": user_prompt}]

    if system_prompt:
        return [ChatMessage.as_system(system_prompt), ChatMessage.as_user(user_content)]
    return [ChatMessage.as_user(user_content)]
