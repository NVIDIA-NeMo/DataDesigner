# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import data_designer.engine.validators.python as python_validator_module
from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.validator_params import CodeValidatorParams
from data_designer.engine.validators.python import (
    PythonLinterMessage,
    PythonLinterMessages,
    PythonValidationStat,
    PythonValidator,
)


@pytest.fixture(scope="module")
def fixture_python_validator():
    return PythonValidator(CodeValidatorParams(code_lang=CodeLang.PYTHON))


@pytest.fixture(scope="module")
def fixture_bad_code_str():
    return "print-it('Hello, world!')\n"


@pytest.fixture(scope="module")
def fixture_good_code_str():
    return """\
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
data = np.random.rand(100, 2)
x = data[:, 0]
y = data[:, 1]

# Plot the data
plt.scatter(x, y)
plt.title("Bandwidth Allocation")
plt.xlabel("Time")
plt.ylabel("Bandwidth Usage")
plt.show()
    """


def test_validate_files_in_path(fixture_python_validator, fixture_bad_code_str, fixture_good_code_str):
    with tempfile.TemporaryDirectory() as temp_dir:
        with open(f"{temp_dir}/good_code.py", "w") as good_code_file:
            good_code_file.write(fixture_good_code_str)
            good_code_file.flush()
            with open(f"{temp_dir}/bad_code.py", "w") as bad_code_file:
                bad_code_file.write(fixture_bad_code_str)
                bad_code_file.flush()

                result = fixture_python_validator.run_validation(
                    [{"code": fixture_bad_code_str}, {"code": fixture_good_code_str}]
                )

                expected_bad_code_messages = [
                    {
                        "symbol": "F821",
                        "message": "Undefined name `it`",
                        "type": "error",
                        "line": 1,
                        "column": 7,
                    }
                ]
                assert result.data[0].is_valid is False
                assert result.data[0].python_linter_score == 0
                assert result.data[0].python_linter_severity == "error"
                assert json.dumps(result.data[0].python_linter_messages, sort_keys=True) == json.dumps(
                    expected_bad_code_messages, sort_keys=True
                )

                assert result.data[1].is_valid is True
                assert result.data[1].python_linter_score == 10.0
                assert result.data[1].python_linter_severity == "none"
                assert result.data[1].python_linter_messages == []


@pytest.mark.parametrize(
    "malformed_code",
    [
        "def f(:\n    return 1\n",
        "x = 1 +\n",
    ],
)
def test_validate_malformed_python(fixture_python_validator: PythonValidator, malformed_code: str) -> None:
    result = fixture_python_validator.run_validation([{"code": malformed_code}])

    assert result.data[0].is_valid is False
    assert result.data[0].python_linter_score == 0
    assert result.data[0].python_linter_severity == "fatal"
    assert result.data[0].python_linter_messages
    assert all(message["type"] == "fatal" for message in result.data[0].python_linter_messages)
    assert all(message["message"] for message in result.data[0].python_linter_messages)


@pytest.mark.parametrize(
    "diagnostic_code",
    [None, "invalid-syntax", "E999", "SyntaxError"],
    ids=["none-code", "ruff-syntax-code", "legacy-ruff-syntax-code", "pylint-syntax-code"],
)
def test_run_linter_classifies_syntax_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, diagnostic_code: str | None
) -> None:
    code_file = tmp_path / "module.py"
    code_file.write_text("pass\n")

    diagnostic = {
        "filename": str(code_file),
        "code": diagnostic_code,
        "location": {"row": 1, "column": 1},
        "message": "Syntax error",
    }

    monkeypatch.setattr(python_validator_module, "find_ruff_bin", lambda: "ruff")
    monkeypatch.setattr(
        python_validator_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps([diagnostic])),
    )

    result = python_validator_module.PythonValidator._run_linter(str(tmp_path))

    message = result["module"].messages[0]
    assert message.type == "fatal"
    assert message.symbol == str(diagnostic_code)


def test_python_linter_validation_stat():
    # trigger division by zero
    assert PythonValidationStat(fatal=0, error=0, warning=1, refactor=0, convention=1, statement=0).score == 0
    # perfect score
    assert PythonValidationStat(fatal=0, error=0, warning=0, refactor=0, convention=0, statement=4).score == 10.0
    # somewhere in between
    assert PythonValidationStat(fatal=0, error=0, warning=2, refactor=0, convention=0, statement=4).score == 5.0


def test_python_linter_messages():
    fatal_msg_1 = PythonLinterMessage(type="fatal", symbol="test", line=12, column=1, message="test")
    fatal_msg_2 = PythonLinterMessage(type="fatal", symbol="test", line=1, column=1, message="test")
    warning_msg = PythonLinterMessage(type="warning", symbol="test", line=10, column=1, message="test")
    refactor_msg = PythonLinterMessage(type="refactor", symbol="test", line=11, column=1, message="test")
    convention_msg = PythonLinterMessage(type="convention", symbol="test", line=34, column=1, message="test")

    python_linter_messages = PythonLinterMessages()
    assert python_linter_messages.is_empty is True
    assert python_linter_messages.severity == "none"
    assert python_linter_messages.is_valid is True

    python_linter_messages.add(refactor_msg)
    python_linter_messages.add(convention_msg)
    python_linter_messages.add(fatal_msg_1)
    python_linter_messages.add(fatal_msg_2)
    python_linter_messages.add(warning_msg)
    assert python_linter_messages.is_empty is False
    assert python_linter_messages.severity == "fatal"
    assert python_linter_messages.is_valid is False

    expected_messages = [
        fatal_msg_2,
        fatal_msg_1,
        warning_msg,
        convention_msg,
        refactor_msg,
    ]
    assert python_linter_messages.messages == expected_messages
