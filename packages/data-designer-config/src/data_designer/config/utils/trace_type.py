# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum


class TraceType(str, Enum):
    """Specifies the type of reasoning trace to capture for LLM columns.

    Traces capture the conversation history during LLM generation, which is
    useful for debugging, analysis, and understanding model behavior.

    Attributes:
        NONE: No trace is captured. This is the default.
        LAST_MESSAGE: Only the final assistant message is captured.
        ALL_MESSAGES: The full conversation history (system/user/assistant/tool)
            is captured.
    """

    NONE = "none"
    LAST_MESSAGE = "last_message"
    ALL_MESSAGES = "all_messages"
