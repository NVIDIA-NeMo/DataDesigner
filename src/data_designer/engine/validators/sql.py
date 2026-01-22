# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.validator_params import CodeValidatorParams
from data_designer.engine.validators.base import BaseValidator, ValidationOutput, ValidationResult
from data_designer.lazy_heavy_imports import pd, sqlfluff

if TYPE_CHECKING:
    import pandas as pd
    import sqlfluff

sqlfluff_logger = logging.getLogger("sqlfluff")
sqlfluff_logger.setLevel(logging.WARNING)


class SQLValidator(BaseValidator):
    def __init__(self, config: CodeValidatorParams):
        self.config = config

    def run_validation(self, data: list[dict]) -> ValidationResult:
        df = pd.DataFrame(data)

        if len(df.columns) > 1:
            raise ValueError("SQL validator assumes single column input")
        target_column = df.columns[0]

        records = df.to_dict(orient="records")

        results = []
        for record in records:
            result = self._validate_query(record[target_column])
            results.append(result)

        return ValidationResult(data=results)

    def _validate_query(self, content: str) -> ValidationResult:
        try:
            result = sqlfluff.lint(
                content,
                dialect=CodeLang.parse_dialect(self.config.code_lang),
            )
            prs_errors = [res for res in result if res["code"].startswith("PRS")]
            error_messages = "\n".join([f"{error['code']}: {error['description']}" for error in prs_errors])
            decimal_pattern = re.compile(r"DECIMAL\(\d+\)")
            decimal_issues = decimal_pattern.findall(content)
            if decimal_issues:
                error_messages += "\nCustom Check: Found DECIMAL definitions without a scale, which may be incorrect."
            if error_messages:
                return ValidationOutput(
                    is_valid=False,
                    error_messages=error_messages,
                )
            return ValidationOutput(is_valid=True, error_messages="")
        except Exception as e:
            return ValidationOutput(
                is_valid=False,
                error_messages=f"Exception during SQL parsing: {e}",
            )
