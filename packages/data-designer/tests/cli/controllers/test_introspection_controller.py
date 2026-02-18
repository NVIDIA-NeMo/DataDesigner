# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click.exceptions
import pytest

from data_designer.cli.controllers.introspection_controller import IntrospectionController

# ---------------------------------------------------------------------------
# show_columns
# ---------------------------------------------------------------------------


def test_show_columns_list_mode(capsys: pytest.CaptureFixture[str]) -> None:
    controller = IntrospectionController()
    controller.show_columns(type_name=None)
    captured = capsys.readouterr()
    assert "llm-text" in captured.out
    assert "sampler" in captured.out


def test_show_columns_specific_type(capsys: pytest.CaptureFixture[str]) -> None:
    controller = IntrospectionController()
    controller.show_columns(type_name="llm-text")
    captured = capsys.readouterr()
    assert "LLMTextColumnConfig" in captured.out


def test_show_columns_all(capsys: pytest.CaptureFixture[str]) -> None:
    controller = IntrospectionController()
    controller.show_columns(type_name="all")
    captured = capsys.readouterr()
    assert "llm-text" in captured.out
    assert "sampler" in captured.out


def test_show_columns_nonexistent_type_exits() -> None:
    controller = IntrospectionController()
    with pytest.raises(click.exceptions.Exit):
        controller.show_columns(type_name="nonexistent_type_xyz")


# ---------------------------------------------------------------------------
# show_samplers
# ---------------------------------------------------------------------------


def test_show_samplers_list(capsys: pytest.CaptureFixture[str]) -> None:
    controller = IntrospectionController()
    controller.show_samplers(type_name=None)
    captured = capsys.readouterr()
    assert "category" in captured.out


def test_show_samplers_specific(capsys: pytest.CaptureFixture[str]) -> None:
    controller = IntrospectionController()
    controller.show_samplers(type_name="category")
    captured = capsys.readouterr()
    assert "sampler_type: category" in captured.out


def test_show_samplers_all_case_insensitive(capsys: pytest.CaptureFixture[str]) -> None:
    controller = IntrospectionController()
    controller.show_samplers(type_name="ALL")
    captured = capsys.readouterr()
    assert "Data Designer Sampler Types Reference" in captured.out
    assert "sampler_type: category" in captured.out


# ---------------------------------------------------------------------------
# show_builder
# ---------------------------------------------------------------------------


def test_show_builder(capsys: pytest.CaptureFixture[str]) -> None:
    controller = IntrospectionController()
    controller.show_builder()
    captured = capsys.readouterr()
    assert "add_column" in captured.out


# ---------------------------------------------------------------------------
# show_sampler_constraints
# ---------------------------------------------------------------------------


def test_show_sampler_constraints(capsys: pytest.CaptureFixture[str]) -> None:
    controller = IntrospectionController()
    controller.show_sampler_constraints()
    captured = capsys.readouterr()
    assert "ScalarInequalityConstraint" in captured.out


# ---------------------------------------------------------------------------
# show_validators
# ---------------------------------------------------------------------------


def test_show_validators_list_text(capsys: pytest.CaptureFixture[str]) -> None:
    controller = IntrospectionController()
    controller.show_validators(type_name=None)
    captured = capsys.readouterr()
    assert "validator_type" in captured.out
    assert "params_class" in captured.out


def test_show_validators_specific_text(capsys: pytest.CaptureFixture[str]) -> None:
    controller = IntrospectionController()
    controller.show_validators(type_name="code")
    captured = capsys.readouterr()
    assert "validator_type: code" in captured.out


def test_show_validators_all_case_insensitive(capsys: pytest.CaptureFixture[str]) -> None:
    controller = IntrospectionController()
    controller.show_validators(type_name="ALL")
    captured = capsys.readouterr()
    assert "Data Designer Validator Types Reference" in captured.out
    assert "validator_type: code" in captured.out


# ---------------------------------------------------------------------------
# show_processors
# ---------------------------------------------------------------------------


def test_show_processors_list_text(capsys: pytest.CaptureFixture[str]) -> None:
    controller = IntrospectionController()
    controller.show_processors(type_name=None)
    captured = capsys.readouterr()
    assert "processor_type" in captured.out
    assert "config_class" in captured.out
