# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from string import Formatter
from typing import Optional

from jinja2 import meta
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import BaseModel
from rich import box
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel

from data_designer.config.column_types import ColumnConfigT, DataDesignerColumnType, column_type_is_llm_generated
from data_designer.config.processors import ProcessorConfig, ProcessorType
from data_designer.config.utils.constants import RICH_CONSOLE_THEME
from data_designer.config.utils.misc import can_run_data_designer_locally
from data_designer.config.validator_params import ValidatorType


class ViolationType(str, Enum):
    ALL_COLUMNS_DROPPED = "all_columns_dropped"
    CODE_COLUMN_MISSING = "code_column_missing"
    CODE_COLUMN_NOT_CODE = "code_column_not_code"
    CODE_LANG_MISMATCH = "code_lang_mismatch"
    EXPRESSION_REFERENCE_MISSING = "expression_reference_missing"
    F_STRING_SYNTAX = "f_string_syntax"
    LOCAL_ONLY_COLUMN = "local_only_column"
    INVALID_COLUMN = "invalid_column"
    INVALID_MODEL_CONFIG = "invalid_model_config"
    INVALID_REFERENCE = "invalid_reference"
    PROMPT_WITHOUT_REFERENCES = "prompt_without_references"


class ViolationLevel(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


class Violation(BaseModel):
    column: Optional[str] = None
    type: ViolationType
    message: str
    level: ViolationLevel

    @property
    def has_column(self) -> bool:
        return self.column is not None


def validate_data_designer_config(
    columns: list[ColumnConfigT],
    processor_configs: list[ProcessorConfig],
    allowed_references: list[str],
) -> list[Violation]:
    violations = []
    violations.extend(validate_prompt_templates(columns=columns, allowed_references=allowed_references))
    violations.extend(validate_code_validation(columns=columns))
    violations.extend(validate_expression_references(columns=columns, allowed_references=allowed_references))
    violations.extend(validate_columns_not_all_dropped(columns=columns))
    violations.extend(validate_drop_columns_processor(columns=columns, processor_configs=processor_configs))
    if not can_run_data_designer_locally():
        violations.extend(validate_local_only_columns(columns=columns))
    return violations


def rich_print_violations(violations: list[Violation]) -> None:
    if len(violations) == 0:
        return

    console = Console(theme=RICH_CONSOLE_THEME)

    render_list = []
    render_list.append(
        Padding(
            Panel(
                f"🔎 Identified {len(violations)} validation "
                f"issue{'' if len(violations) == 1 else 's'} "
                "in your Data Designer column definitions",
                box=box.SIMPLE,
                highlight=True,
            ),
            (0, 0, 1, 0),
        )
    )

    for v in violations:
        emoji = "🛑" if v.level == ViolationLevel.ERROR else "⚠️"

        error_title = f"{emoji} {v.level.upper()} | {v.type.value.upper()}"

        render_list.append(
            Padding(
                Panel(
                    f"{error_title}\n\n{v.message}",
                    box=box.HORIZONTALS,
                    title=f"Column: {v.column}" if v.has_column else "",
                    padding=(1, 0, 1, 1),
                    highlight=True,
                ),
                (0, 0, 1, 0),
            )
        )

    console.print(Group(*render_list), markup=False)


def validate_prompt_templates(
    columns: list[ColumnConfigT],
    allowed_references: list[str],
) -> list[Violation]:
    env = ImmutableSandboxedEnvironment()

    columns_with_prompts = [c for c in columns if column_type_is_llm_generated(c.column_type)]

    violations = []
    for column in columns_with_prompts:
        for prompt_type in ["prompt", "system_prompt"]:
            if not hasattr(column, prompt_type) or getattr(column, prompt_type) is None:
                continue

            prompt = getattr(column, prompt_type)

            # check for invalid references
            prompt_references = set()
            prompt_references.update(meta.find_undeclared_variables(env.parse(prompt)))
            invalid_references = list(set(prompt_references) - set(allowed_references))
            num_invalid = len(invalid_references)
            if num_invalid > 0:
                ref_msg = (
                    f"references {num_invalid} columns that do not exist"
                    if num_invalid > 1
                    else "references a column that does not exist"
                )
                invalid_references = ", ".join([f"'{r}'" for r in invalid_references])
                message = f"The {prompt_type} template for '{column.name}' {ref_msg}: {invalid_references}."
                violations.append(
                    Violation(
                        column=column.name,
                        type=ViolationType.INVALID_REFERENCE,
                        message=message,
                        level=ViolationLevel.ERROR,
                    )
                )

            # check for prompts without references

            if (
                prompt_type == "prompt"
                and len(prompt_references) == 0
                and (not hasattr(column, "multi_modal_context") or getattr(column, "multi_modal_context") is None)
            ):
                message = (
                    f"The {prompt_type} template for '{column.name}' does not reference any columns. "
                    "This means the same prompt will be used for every row in the dataset. To increase "
                    "the diversity of the generated data, consider adding references to other columns "
                    "in the prompt template."
                )
                violations.append(
                    Violation(
                        column=column.name,
                        type=ViolationType.PROMPT_WITHOUT_REFERENCES,
                        message=message,
                        level=ViolationLevel.WARNING,
                    )
                )

            # check for f-string syntax
            f_string_references = _get_string_formatter_references(prompt, allowed_references)
            if len(f_string_references) > 0:
                f_string_references = ", ".join([f"'{r}'" for r in f_string_references])
                message = (
                    f"The {prompt_type} template for '{column.name}' references the "
                    f"following columns using f-string syntax: {f_string_references}. "
                    "Please use jinja2 syntax to reference columns: {reference} -> {{ reference }}."
                )
                violations.append(
                    Violation(
                        column=column.name,
                        type=ViolationType.F_STRING_SYNTAX,
                        message=message,
                        level=ViolationLevel.WARNING,
                    )
                )
    return violations


def validate_code_validation(
    columns: list[ColumnConfigT],
) -> list[Violation]:
    columns_by_name = {c.name: c for c in columns}
    code_validation_columns = [
        c for c in columns if c.column_type == DataDesignerColumnType.VALIDATION and c.validator_type == "code"
    ]

    violations = []
    for validation_column in code_validation_columns:
        for target_column_name in validation_column.target_columns:
            # check that the target column exists
            if target_column_name not in columns_by_name:
                message = f"Target code column '{target_column_name}' not found in column list."
                violations.append(
                    Violation(
                        column=validation_column.name,
                        type=ViolationType.CODE_COLUMN_MISSING,
                        message=message,
                        level=ViolationLevel.ERROR,
                    )
                )
                continue

            # check for consistent code languages
            target_column = columns_by_name[target_column_name]
            if target_column.column_type != DataDesignerColumnType.LLM_CODE:
                message = (
                    f"Code validation column '{validation_column.name}' is set to validate "
                    f"code, but the target column was generated as {target_column.column_type}."
                )
                violations.append(
                    Violation(
                        column=validation_column.name,
                        type=ViolationType.CODE_COLUMN_NOT_CODE,
                        message=message,
                        level=ViolationLevel.WARNING,
                    )
                )
            elif target_column.code_lang != validation_column.validator_params.code_lang:
                message = (
                    f"Code validation column '{validation_column.name}' is set to validate "
                    f"{validation_column.validator_params.code_lang}, but the target column was generated as "
                    f"{target_column.code_lang}."
                )
                violations.append(
                    Violation(
                        column=validation_column.name,
                        type=ViolationType.CODE_LANG_MISMATCH,
                        message=message,
                        level=ViolationLevel.ERROR,
                    )
                )

    return violations


def validate_columns_not_all_dropped(
    columns: list[ColumnConfigT],
) -> list[Violation]:
    remaining_cols = [c for c in columns if c.column_type != DataDesignerColumnType.SEED_DATASET and not c.drop]

    if len(remaining_cols) == 0:
        return [
            Violation(
                column=None,
                type=ViolationType.ALL_COLUMNS_DROPPED,
                message=(
                    "All generated columns are configured to be dropped. "
                    "Please mark at least one column with `drop=False`."
                ),
                level=ViolationLevel.ERROR,
            )
        ]

    return []


def validate_drop_columns_processor(
    columns: list[ColumnConfigT],
    processor_configs: list[ProcessorConfig],
) -> list[Violation]:
    all_column_names = set([c.name for c in columns])
    for processor_config in processor_configs:
        if processor_config.processor_type == ProcessorType.DROP_COLUMNS:
            invalid_columns = set(processor_config.column_names) - all_column_names
            if len(invalid_columns) > 0:
                return [
                    Violation(
                        column=c,
                        type=ViolationType.INVALID_COLUMN,
                        message=f"Drop columns processor is configured to drop column '{c!r}', but the column is not defined.",
                        level=ViolationLevel.ERROR,
                    )
                    for c in invalid_columns
                ]
    return []


def validate_expression_references(
    columns: list[ColumnConfigT],
    allowed_references: list[str],
) -> list[Violation]:
    expression_columns = [c for c in columns if c.column_type == DataDesignerColumnType.EXPRESSION]
    violations = []
    for expression_column in expression_columns:
        for reference in expression_column.required_columns:
            if reference not in allowed_references:
                violations.append(
                    Violation(
                        column=expression_column.name,
                        type=ViolationType.EXPRESSION_REFERENCE_MISSING,
                        message=f"Expression column '{expression_column.name}' references missing column '{reference}'.",
                        level=ViolationLevel.ERROR,
                    )
                )
    return violations


def validate_local_only_columns(
    columns: list[ColumnConfigT],
) -> list[Violation]:
    violations = []
    validation_columns = [c for c in columns if c.column_type == DataDesignerColumnType.VALIDATION]

    # Local validation columns
    for validation_column in validation_columns:
        if validation_column.validator_type == ValidatorType.LOCAL_CALLABLE:
            violations.append(
                Violation(
                    column=validation_column.name,
                    type=ViolationType.LOCAL_ONLY_COLUMN,
                    message="Validation using functions are only supported when running Data Designer locally",
                    level=ViolationLevel.ERROR,
                )
            )
    return violations


def _get_string_formatter_references(template: str, allowed_references: list[str]) -> list[str]:
    return [
        k[1].strip()
        for k in Formatter().parse(template)
        if len(k) > 1 and k[1] is not None and k[1].strip() in allowed_references
    ]
