# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.validator_params import CodeValidatorParams
from data_designer.engine.validators.sql import SQLValidator


def test_valid_ansi_sql_code() -> None:
    sql_validator = SQLValidator(CodeValidatorParams(code_lang=CodeLang.SQL_ANSI))
    code = "SELECT category, COUNT(*) as total_incidents FROM security_incidents_2 GROUP BY category;"
    result = sql_validator.run_validation([{"sql": code}])
    assert result.data[0].is_valid
    assert result.data[0].error_messages == ""


def test_invalid_ansi_sql_code() -> None:
    sql_validator = SQLValidator(CodeValidatorParams(code_lang=CodeLang.SQL_ANSI))
    code = "NOT SQL"
    result = sql_validator.run_validation([{"sql": code}])
    assert not result.data[0].is_valid
    assert result.data[0].error_messages == "PRS: Line 1, Position 1: Found unparsable section: 'NOT SQL'"


def test_sql_validator_multi_column_input_raises() -> None:
    sql_validator = SQLValidator(CodeValidatorParams(code_lang=CodeLang.SQL_ANSI))
    with pytest.raises(ValueError, match="single column input"):
        sql_validator.run_validation([{"sql": "SELECT 1", "extra": "ignored"}])


def test_sql_validator_decimal_without_scale_fails() -> None:
    sql_validator = SQLValidator(CodeValidatorParams(code_lang=CodeLang.SQL_ANSI))
    code = "CREATE TABLE example (amount DECIMAL(10));"
    result = sql_validator.run_validation([{"sql": code}])
    assert not result.data[0].is_valid
    assert "DECIMAL definitions without a scale" in result.data[0].error_messages


def test_sql_validator_handles_lint_exception() -> None:
    sql_validator = SQLValidator(CodeValidatorParams(code_lang=CodeLang.SQL_ANSI))
    with patch("data_designer.lazy_heavy_imports.sqlfluff.lint", side_effect=RuntimeError("boom")):
        result = sql_validator.run_validation([{"sql": "SELECT 1"}])
    assert not result.data[0].is_valid
    assert "Exception during SQL parsing" in result.data[0].error_messages
