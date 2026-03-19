# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_designer.engine.resources.agent_rollout.types import AgentRolloutSeedParseError
from data_designer.engine.resources.agent_rollout.utils import (
    build_message,
    load_jsonl_rows,
    normalize_message_role,
    require_string,
)


def test_normalize_message_role_maps_developer_to_system() -> None:
    assert normalize_message_role("developer", context="test") == "system"


def test_normalize_message_role_rejects_unknown_role() -> None:
    with pytest.raises(AgentRolloutSeedParseError, match="Unsupported message role"):
        normalize_message_role("moderator", context="test")


def test_build_message_normalizes_via_chat_message() -> None:
    result = build_message(role="user", content="hello")
    assert result["role"] == "user"
    assert isinstance(result["content"], list)


def test_load_jsonl_rows_rejects_invalid_json_and_non_objects(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text("not json\n", encoding="utf-8")
    with pytest.raises(AgentRolloutSeedParseError, match="Invalid JSON"):
        load_jsonl_rows(bad_json)

    non_object = tmp_path / "array.jsonl"
    non_object.write_text(json.dumps([1, 2, 3]) + "\n", encoding="utf-8")
    with pytest.raises(AgentRolloutSeedParseError, match="Expected JSON object"):
        load_jsonl_rows(non_object)


def test_require_string_rejects_empty_values() -> None:
    with pytest.raises(AgentRolloutSeedParseError, match="Expected non-empty string"):
        require_string("", "test field")
    with pytest.raises(AgentRolloutSeedParseError, match="Expected non-empty string"):
        require_string(None, "test field")
