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
        "families": [{"family": "columns", "count": 1, "files": ["data_designer/config/column_configs.py"]}],
        "types": {
            "columns": [{"type": "a", "description": "A thing."}],
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
    assert "config_root: /some/path/to/config" in result
    assert "Do not search other modules" in result
    assert "builder: {config_root}/config_builder.py" in result
    assert "## Types" in result
    assert "## Commands" in result


def test_format_context_text_no_usable_aliases_shows_warning() -> None:
    data: dict[str, Any] = {
        "library_version": "1.0.0",
        "config_module_path": "/some/path/to/config",
        "config_builder_file": "data_designer/config/config_builder.py",
        "base_config_file": "data_designer/config/base.py",
        "families": [],
        "types": {},
        "state": {
            "model_aliases": {
                "default_provider": "nvidia",
                "items": [{"model_alias": "bad", "usable": False, "reason": "missing key"}],
            },
            "persona_datasets": {"items": []},
        },
        "operations": [],
    }
    result = format_context_text(data)

    assert "No usable model aliases" in result
    assert "Tell the user" in result


# --- format_types_text ---


def test_format_types_text_single_family_with_config_root() -> None:
    data: dict[str, Any] = {
        "config_module_path": "/some/path/to/data_designer/config",
        "family": "columns",
        "files": ["data_designer/config/column_configs.py"],
        "items": [
            {"type": "alpha", "description": "Alpha desc."},
            {"type": "beta", "description": "Beta desc."},
        ],
    }
    result = format_types_text(data)

    assert "config_root: /some/path/to/data_designer/config" in result
    assert "### columns" in result
    assert "file: {config_root}/column_configs.py" in result
    assert "alpha" in result
    assert "Alpha desc." in result


def test_format_types_text_all_families_with_config_root() -> None:
    data: dict[str, Any] = {
        "config_module_path": "/some/path/to/data_designer/config",
        "families": [
            {"family": "columns", "count": 1, "files": ["data_designer/config/column_configs.py"]},
            {"family": "samplers", "count": 1, "files": ["data_designer/config/sampler_params.py"]},
        ],
        "items": {
            "columns": [{"type": "a", "description": "Desc A."}],
            "samplers": [{"type": "b", "description": "Desc B."}],
        },
    }
    result = format_types_text(data)

    assert "config_root: /some/path/to/data_designer/config" in result
    assert "### columns" in result
    assert "file: {config_root}/column_configs.py" in result
    assert "file: {config_root}/sampler_params.py" in result


def test_format_types_text_empty_items() -> None:
    data: dict[str, Any] = {"family": "columns", "files": ["data_designer/config/column_configs.py"], "items": []}
    result = format_types_text(data)

    assert "file: {config_root}/column_configs.py" in result
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
