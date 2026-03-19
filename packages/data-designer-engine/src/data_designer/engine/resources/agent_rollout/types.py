# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


class AgentRolloutSeedParseError(ValueError): ...


@dataclass
class NormalizedAgentRolloutRecord:
    trace_id: str
    source_kind: str
    source_path: str
    root_session_id: str
    agent_id: str | None
    is_sidechain: bool
    cwd: str | None
    project_path: str | None
    git_branch: str | None
    started_at: str | None
    ended_at: str | None
    messages: list[dict[str, Any]]
    source_meta: dict[str, Any]
    # Derived fields (computed in __post_init__)
    message_count: int = field(init=False)
    tool_call_count: int = field(init=False)
    final_assistant_message: str | None = field(init=False)

    def __post_init__(self) -> None:
        self.message_count = len(self.messages)
        self.tool_call_count = sum(len(m.get("tool_calls", [])) for m in self.messages)
        self.final_assistant_message = _extract_final_assistant_text(self.messages)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _extract_final_assistant_text(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if content is None:
            continue
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if "text" in block:
                        text = block["text"]
                        if isinstance(text, str) and text:
                            text_parts.append(text)
                elif isinstance(block, str) and block:
                    text_parts.append(block)
            joined = "\n\n".join(text_parts)
            if joined:
                return joined
        elif isinstance(content, str) and content:
            return content
    return None
