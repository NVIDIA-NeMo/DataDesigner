# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.validator_params import CodeValidatorParams
from data_designer.engine.validators.sql import SQLValidator


def test_valid_ansi_sql_code():
    sql_validator = SQLValidator(CodeValidatorParams(code_lang=CodeLang.SQL_ANSI))
    code = "SELECT category, COUNT(*) as total_incidents FROM security_incidents_2 GROUP BY category;"
    result = sql_validator.run_validation([{"sql": code}])
    assert result.data[0].is_valid
    assert result.data[0].error_messages == ""


def test_invalid_ansi_sql_code():
    sql_validator = SQLValidator(CodeValidatorParams(code_lang=CodeLang.SQL_ANSI))
    code = "NOT SQL"
    result = sql_validator.run_validation([{"sql": code}])
    assert not result.data[0].is_valid
    assert result.data[0].error_messages == "PRS: Line 1, Position 1: Found unparsable section: 'NOT SQL'"
