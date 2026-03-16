# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from data_designer.cli.utils.agent_text_formatter import (
    format_context_text,
    format_model_aliases_text,
    format_persona_datasets_text,
    format_types_text,
)

# --- format_context_text ---


def test_format_context_text_includes_config_module_path() -> None:
    data: dict[str, Any] = {
        "library_version": "1.0.0",
        "config_module_path": "/some/path/to/config",
        "config_builder_file": "data_designer/config/config_builder.py",
        "base_config_file": "data_designer/config/base.py",
        "families": [{"family": "columns", "count": 1, "files": ["/a.py"]}],
        "types": {
            "columns": [{"type_name": "a", "description": "A thing."}],
        },
        "state": {
            "model_aliases": {"default_provider": None, "items": []},
            "persona_datasets": {"items": []},
        },
        "operations": [{"command_pattern": "agent context", "description": "Bootstrap payload."}],
    }
    result = format_context_text(data)

    assert "Data Designer v1.0.0" in result
    assert "## Config Module" in result
    assert "root: /some/path/to/config" in result
    assert "builder: config_builder.py" in result
    assert "## Types" in result
    assert "## Commands" in result


# --- format_types_text ---


def test_format_types_text_single_family_shows_file_above_table() -> None:
    data: dict[str, Any] = {
        "family": "columns",
        "files": ["/path/to/column_configs.py"],
        "items": [
            {"type_name": "alpha", "description": "Alpha desc."},
            {"type_name": "beta", "description": "Beta desc."},
        ],
    }
    result = format_types_text(data)

    assert "### columns" in result
    assert "file: /path/to/column_configs.py" in result
    assert "alpha" in result
    assert "Alpha desc." in result


def test_format_types_text_all_families_shows_file_per_family() -> None:
    data: dict[str, Any] = {
        "families": [
            {"family": "columns", "count": 1, "files": ["/cols.py"]},
            {"family": "samplers", "count": 1, "files": ["/samp.py"]},
        ],
        "items": {
            "columns": [{"type_name": "a", "description": "Desc A."}],
            "samplers": [{"type_name": "b", "description": "Desc B."}],
        },
    }
    result = format_types_text(data)

    assert "### columns" in result
    assert "file: /cols.py" in result
    assert "file: /samp.py" in result


def test_format_types_text_empty_items() -> None:
    data: dict[str, Any] = {"family": "columns", "files": ["/cols.py"], "items": []}
    result = format_types_text(data)

    assert "(no items)" in result


# --- format_model_aliases_text ---


def test_format_model_aliases_text_with_items() -> None:
    state: dict[str, Any] = {
        "default_provider": "nvidia",
        "items": [
            {
                "model_alias": "test",
                "model": "meta/llama-3",
                "generation_type": "chat",
                "effective_provider": "nvidia",
                "usable": True,
                "reason": None,
            },
        ],
    }
    result = format_model_aliases_text(state)

    assert "default_provider: nvidia" in result
    assert "test" in result
    assert "meta/llama-3" in result


def test_format_model_aliases_text_empty() -> None:
    state: dict[str, Any] = {"default_provider": None, "items": []}
    result = format_model_aliases_text(state)

    assert "default_provider: (none)" in result
    assert "(no items)" in result


# --- format_persona_datasets_text ---


def test_format_persona_datasets_text() -> None:
    state: dict[str, Any] = {
        "items": [{"locale": "en_US", "size": "10MB", "installed": True}],
    }
    result = format_persona_datasets_text(state)

    assert "en_US" in result
    assert "True" in result
