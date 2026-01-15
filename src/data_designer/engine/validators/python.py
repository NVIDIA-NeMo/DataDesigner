# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import logging
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel
from ruff.__main__ import find_ruff_bin

from data_designer.config.validator_params import CodeValidatorParams
from data_designer.engine.validators.base import BaseValidator, ValidationOutput, ValidationResult
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

PYLINT_ERROR_CATEGORIES_ORDERED = [
    "fatal",
    "error",
    "warning",
    "convention",
    "refactor",
]
PYLINT_VALID_LEVELS = {"none", "warning", "convention", "refactor"}

TYPE_FROM_SYMBOL = {
    "E": "refactor",
    "F": "error",
    "SIM": "refactor",
    "PLC": "convention",
    "PLE": "error",
    "PLR": "refactor",
    "PLW": "warning",
    "SyntaxError": "fatal",
}

PYTHON_MESSAGES_FIELD = "python_linter_messages"
RECORD_ID_COLUMN_NAME = "internal_code_record_id"


class PythonValidationStat(BaseModel):
    fatal: int = 0
    error: int = 0
    warning: int = 0
    refactor: int = 0
    convention: int = 0
    statement: int = 0

    @property
    def score(self) -> float:
        # https://pylint.pycqa.org/en/latest/user_guide/configuration/all-options.html#evaluation
        if self.statement == 0:  # prevent division by zero down below
            self.statement = max(1, self.statement)
        return max(
            0,
            (
                0
                if self.fatal
                else 10.0
                - ((float(5 * self.error + self.warning + self.refactor + self.convention) / self.statement) * 10)
            ),
        )


class PythonLinterMessage(BaseModel):
    type: str
    symbol: str
    line: int
    column: int
    message: str

    @property
    def type_sort_order(self) -> int:
        return PYLINT_ERROR_CATEGORIES_ORDERED.index(self.type)


class PythonLinterMessages(BaseModel):
    _messages: list[PythonLinterMessage] = []

    @property
    def messages(self) -> list[PythonLinterMessage]:
        # Ordered by severity first then by line number
        return sorted(self._messages, key=lambda msg: (msg.type_sort_order, msg.line))

    def add(self, message: PythonLinterMessage) -> None:
        self._messages.append(message)

    def get_count_by_type(self) -> dict[str, int]:
        count_by_type = defaultdict(int)
        for message in self.messages:
            count_by_type[message.type] += 1
        return dict(count_by_type)

    @property
    def is_empty(self) -> bool:
        return len(self.messages) == 0

    @property
    def severity(self) -> str:
        if self.is_empty:
            return "none"
        return self.messages[0].type

    @property
    def is_valid(self) -> bool:
        return self.is_empty or self.messages[0].type in PYLINT_VALID_LEVELS


class PythonValidator(BaseValidator):
    def __init__(self, config: CodeValidatorParams):
        self.config = config

    def run_validation(self, data: list[dict]) -> ValidationResult:
        df = pd.DataFrame(data)

        if len(df.columns) > 1:
            raise ValueError("Python validator assumes single column input")
        target_column = df.columns[0]

        df.loc[:, RECORD_ID_COLUMN_NAME] = [uuid4() for _ in range(df.shape[0])]
        with tempfile.TemporaryDirectory() as temp_dir:
            _ = df.apply(
                self._write_code_to_file,
                args=(target_column, temp_dir),
                axis=1,
            )
            results = self._validate_files_in_path(path=temp_dir)

            records = df.to_dict(orient="records")

            ordered_results = []
            for record in records:
                module_id = self._get_module_name(record[RECORD_ID_COLUMN_NAME], target_column)
                result = results.get(module_id)
                if result is not None:
                    ordered_results.append(result)

        return ValidationResult(data=ordered_results)

    def _validate_files_in_path(self, path: str) -> dict[str, ValidationOutput]:
        lint_results = self._run_linter(path)

        scores_by_module = self._get_scores(
            {
                module: messages.get_count_by_type()
                | {"statement": self._count_python_statements(f"{path}/{module}.py")}
                for module, messages in lint_results.items()
            }
        )

        validation_result = {}
        for module, score in scores_by_module.items():
            messages = lint_results.get(module, PythonLinterMessages())
            metadata = {
                "python_linter_score": score,
                "python_linter_severity": messages.severity,
                PYTHON_MESSAGES_FIELD: [m.model_dump() for m in messages.messages],
            }
            validation_result[module] = ValidationOutput(is_valid=messages.is_valid, **metadata)
        return validation_result

    def _write_code_to_file(self, row: pd.Series, code_column: str, path: str) -> None:
        with open(f"{path}/{self._get_module_name(row[RECORD_ID_COLUMN_NAME], code_column)}.py", "w") as file:
            file.write(row[code_column])

    @staticmethod
    def _get_module_name(record_id: str, column_name: str) -> str:
        return f"{record_id}_{column_name}"

    @staticmethod
    def _run_linter(codebase_path: str) -> dict[str, PythonLinterMessages]:
        # Create empty dict for output
        processed = {}
        for file in Path(codebase_path).glob("*.py"):
            processed[file.stem] = PythonLinterMessages()

        # Run ruff linter with JSON output
        ruff_bin = find_ruff_bin()

        ruff_exec = subprocess.run(
            [
                ruff_bin,
                "check",
                "--select",
                "E,F6,F7,F8,SIM,PLC,PLE,PLR,PLW",
                "--output-format=json",
                codebase_path,
            ],
            text=True,
            capture_output=True,
            check=False,
            cwd=Path.cwd(),
        )
        ruff_output = ruff_exec.stdout

        # Parse JSON output
        try:
            diagnostics = json.loads(ruff_output)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse ruff JSON output: {e}")

        if not diagnostics:
            return processed  # no errors or warnings

        for diagnostic in diagnostics:
            filename = diagnostic["filename"]
            code = diagnostic["code"]
            location = diagnostic["location"]
            message = diagnostic["message"]

            # Extract alphabetic prefix from code for type mapping
            alpha_prefix = "".join(c for c in code if c.isalpha())
            error_type = TYPE_FROM_SYMBOL.get(alpha_prefix, "warning")

            processed[Path(filename).stem].add(
                PythonLinterMessage(
                    type=error_type,
                    symbol=code,
                    line=location["row"],
                    column=location["column"],
                    message=message,
                )
            )

        return processed

    @staticmethod
    def _get_scores(stats_by_module: dict[str, dict[str, int]]) -> dict[str, float]:
        scores = {}
        for key, item in stats_by_module.items():
            stat = PythonValidationStat(**item)
            scores[key] = stat.score
        return scores

    @staticmethod
    def _count_python_statements(file_path: str) -> int:
        """Count the number of statements in a Python file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            return sum(1 for node in ast.walk(tree) if isinstance(node, ast.stmt))
        except Exception:
            return 0
