# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from data_designer.cli.utils.agent_introspection import get_family_schema
from data_designer.cli.utils.agent_text_formatter import (
    format_builder_text,
    format_context_text,
    format_model_aliases_text,
    format_persona_datasets_text,
    format_schema_text,
    format_types_text,
)

# --- format_context_text ---


def test_format_context_text_includes_builder_section() -> None:
    data: dict[str, Any] = {
        "families": [{"family": "columns", "count": 1}],
        "types": {
            "columns": [{"type_name": "a", "class_name": "A", "import_path": "m.A"}],
        },
        "state": {
            "model_aliases": {"default_provider": None, "items": []},
            "persona_datasets": {"items": []},
        },
        "builder": {
            "class_name": "DataDesignerConfigBuilder",
            "import_path": "data_designer.config.DataDesignerConfigBuilder",
            "methods": [{"name": "add_column", "signature": "add_column(col)", "summary": "Add a column."}],
        },
        "operations": [{"command_pattern": "agent context", "description": "Bootstrap payload."}],
    }
    result = format_context_text(data)

    assert "## Builder" in result
    assert "DataDesignerConfigBuilder:" in result
    assert "add_column(col)" in result


# --- format_types_text ---


def test_format_types_text_single_family() -> None:
    data: dict[str, Any] = {
        "family": "columns",
        "items": [
            {"type_name": "alpha", "class_name": "AlphaConfig", "import_path": "mod.AlphaConfig"},
            {"type_name": "beta", "class_name": "BetaConfig", "import_path": "mod.BetaConfig"},
        ],
    }
    result = format_types_text(data)

    assert "# columns" in result
    assert "alpha" in result
    assert "AlphaConfig" in result


def test_format_types_text_all_families() -> None:
    data: dict[str, Any] = {
        "families": [{"family": "columns", "count": 2}],
        "items": {
            "columns": [
                {"type_name": "a", "class_name": "A", "import_path": "m.A"},
                {"type_name": "b", "class_name": "B", "import_path": "m.B"},
            ],
        },
    }
    result = format_types_text(data)

    assert "columns: 2 types" in result
    assert "a" in result
    assert "b" in result


def test_format_types_text_empty_items() -> None:
    data: dict[str, Any] = {"family": "columns", "items": []}
    result = format_types_text(data)

    assert "(no items)" in result


# --- format_schema_text ---


def test_format_schema_text_single_type() -> None:
    data: dict[str, Any] = {
        "type_name": "llm-text",
        "class_name": "LLMTextColumnConfig",
        "schema_text": "LLMTextColumnConfig:\n  name: str  [required]",
    }
    result = format_schema_text(data)

    assert "LLMTextColumnConfig:" in result
    assert "name: str  [required]" in result


def test_format_schema_text_all_types() -> None:
    data: dict[str, Any] = {
        "family": "columns",
        "items": [
            {"type_name": "a", "class_name": "A", "schema_text": "A:\n  x: int  [required]"},
            {"type_name": "b", "class_name": "B", "schema_text": "B:\n  y: str = 'hi'"},
        ],
    }
    result = format_schema_text(data)

    assert "# columns schemas (2 types)" in result
    assert "A:\n  x: int  [required]" in result
    assert "B:\n  y: str = 'hi'" in result


def test_format_schema_text_passes_through_schema_text() -> None:
    schema_text = "TestModel:\n  name: str  [required]\n  count: int = 0"
    data: dict[str, Any] = {"type_name": "test", "class_name": "TestModel", "schema_text": schema_text}
    result = format_schema_text(data)

    assert result == schema_text


# --- format_builder_text ---


def test_format_builder_text_renders_methods() -> None:
    data: dict[str, Any] = {
        "class_name": "MyBuilder",
        "import_path": "data_designer.config.MyBuilder",
        "methods": [
            {"name": "add_column", "signature": "add_column(column: ColumnConfig)", "summary": "Add a column."},
            {"name": "build", "signature": "build()", "summary": "Build the config."},
        ],
    }
    result = format_builder_text(data)

    assert "MyBuilder:" in result
    assert "usage: dd.MyBuilder" in result
    assert "add_column(column: ColumnConfig)" in result
    assert "Add a column." in result


def test_format_builder_text_handles_method_without_summary() -> None:
    data: dict[str, Any] = {
        "class_name": "Builder",
        "import_path": "mod.Builder",
        "methods": [{"name": "reset", "signature": "reset()", "summary": None}],
    }
    result = format_builder_text(data)

    assert "reset()" in result


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

    assert "# persona datasets" in result
    assert "en_US" in result
    assert "True" in result


# --- Real config models ---


@pytest.mark.parametrize(
    "family,type_name",
    [
        ("columns", "llm-text"),
        ("columns", "sampler"),
        ("samplers", "category"),
        ("validators", "code"),
        ("constraints", "scalar_inequality"),
    ],
    ids=["columns-llm-text", "columns-sampler", "samplers-category", "validators-code", "constraints-scalar"],
)
def test_format_schema_text_on_real_config_models(family: str, type_name: str) -> None:
    schema_data = get_family_schema(family, type_name)
    result = format_schema_text(schema_data)

    assert schema_data["class_name"] in result
    assert result == schema_data["schema_text"]


def test_real_column_schema_excludes_discriminator_and_includes_example() -> None:
    schema_data = get_family_schema("columns", "llm-code")
    text = schema_data["schema_text"]

    assert "column_type:" not in text
    assert "allow_resize" not in text
    assert "Example: dd.LLMCodeColumnConfig(" in text
    assert "values:" in text


def test_real_judge_schema_expands_score_and_shows_enum_values() -> None:
    schema_data = get_family_schema("columns", "llm-judge")
    text = schema_data["schema_text"]

    assert "Score:" in text
    assert "name: str  [required]" in text
    assert "options: dict  [required]" in text
    assert "values: none, last_message, all_messages" in text
    assert "Example: dd.LLMJudgeColumnConfig(" in text
    assert "column_type:" not in text
