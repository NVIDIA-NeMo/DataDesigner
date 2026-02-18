# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typer.testing import CliRunner

from data_designer.cli.main import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------


def test_inspect_help() -> None:
    result = runner.invoke(app, ["inspect", "--help"])
    assert result.exit_code == 0
    assert "column" in result.output


# ---------------------------------------------------------------------------
# columns
# ---------------------------------------------------------------------------


def test_columns_no_arg_fails() -> None:
    result = runner.invoke(app, ["inspect", "column"])
    assert result.exit_code != 0


def test_columns_specific_type() -> None:
    result = runner.invoke(app, ["inspect", "column", "llm-text"])
    assert result.exit_code == 0
    assert "LLMTextColumnConfig" in result.output


def test_columns_nonexistent_exits_with_error() -> None:
    result = runner.invoke(app, ["inspect", "column", "nonexistent"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------


def test_samplers_specific() -> None:
    result = runner.invoke(app, ["inspect", "sampler", "category"])
    assert result.exit_code == 0
    assert "sampler_type: category" in result.output


def test_samplers_all_case_insensitive() -> None:
    result = runner.invoke(app, ["inspect", "sampler", "ALL"])
    assert result.exit_code == 0
    assert "Data Designer Sampler Types Reference" in result.output
    assert "sampler_type: category" in result.output


def test_samplers_no_arg_fails() -> None:
    result = runner.invoke(app, ["inspect", "sampler"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# validators
# ---------------------------------------------------------------------------


def test_validators_no_arg_fails() -> None:
    result = runner.invoke(app, ["inspect", "validator"])
    assert result.exit_code != 0


def test_validators_specific() -> None:
    result = runner.invoke(app, ["inspect", "validator", "code"])
    assert result.exit_code == 0
    assert "validator_type: code" in result.output


def test_validators_all_case_insensitive() -> None:
    result = runner.invoke(app, ["inspect", "validator", "ALL"])
    assert result.exit_code == 0
    assert "Data Designer Validator Types Reference" in result.output
    assert "validator_type: code" in result.output


# ---------------------------------------------------------------------------
# processors
# ---------------------------------------------------------------------------


def test_processors_no_arg_fails() -> None:
    result = runner.invoke(app, ["inspect", "processor"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# config_builder
# ---------------------------------------------------------------------------


def test_config_builder() -> None:
    result = runner.invoke(app, ["inspect", "config_builder"])
    assert result.exit_code == 0
    assert "add_column" in result.output
    assert "DataDesignerConfigBuilder" in result.output
    assert "Parameters:" in result.output


# ---------------------------------------------------------------------------
# constraints
# ---------------------------------------------------------------------------


def test_constraints() -> None:
    result = runner.invoke(app, ["inspect", "sampler-constraints"])
    assert result.exit_code == 0
    output = result.output
    assert "ScalarInequalityConstraint" in output or "InequalityOperator" in output


# ---------------------------------------------------------------------------
# import hints
# ---------------------------------------------------------------------------


def test_import_hint_shown_in_text_output() -> None:
    result = runner.invoke(app, ["inspect", "column", "llm-text"])
    assert result.exit_code == 0
    assert "import data_designer.config as dd" in result.output
    assert "dd.LLMTextColumnConfig" in result.output


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_help() -> None:
    result = runner.invoke(app, ["list", "--help"])
    assert result.exit_code == 0
    for subcmd in ("model-aliases", "persona-datasets", "columns", "samplers", "validators", "processors"):
        assert subcmd in result.output


def test_list_model_aliases() -> None:
    result = runner.invoke(app, ["list", "model-aliases"])
    assert result.exit_code == 0


def test_list_persona_datasets() -> None:
    result = runner.invoke(app, ["list", "persona-datasets"])
    assert result.exit_code == 0
    assert "Nemotron-Personas Datasets" in result.output


def test_list_column_types() -> None:
    result = runner.invoke(app, ["list", "columns"])
    assert result.exit_code == 0
    assert "llm-text" in result.output


def test_list_sampler_types() -> None:
    result = runner.invoke(app, ["list", "samplers"])
    assert result.exit_code == 0
    assert "category" in result.output


def test_list_validator_types() -> None:
    result = runner.invoke(app, ["list", "validators"])
    assert result.exit_code == 0
    assert "code" in result.output


def test_list_processor_types() -> None:
    result = runner.invoke(app, ["list", "processors"])
    assert result.exit_code == 0
