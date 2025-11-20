# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

from data_designer.config.column_configs import (
    ExpressionColumnConfig,
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    Score,
    ValidationColumnConfig,
)
from data_designer.config.dataset_builders import BuildStage
from data_designer.config.models import ImageContext, ModalityDataType
from data_designer.config.processors import DropColumnsProcessorConfig
from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.utils.validation import (
    Violation,
    ViolationLevel,
    ViolationType,
    rich_print_violations,
    validate_code_validation,
    validate_columns_not_all_dropped,
    validate_data_designer_config,
    validate_expression_references,
    validate_prompt_templates,
)
from data_designer.config.validator_params import CodeValidatorParams

STUB_MODEL_ALIAS = "stub-alias"


VALID_COLUMNS = [
    SamplerColumnConfig(
        name="random_number",
        sampler_type="uniform",
        params={"low": 0, "high": 10},
    ),
    LLMTextColumnConfig(
        name="valid_reference",
        prompt="Why is {{ random_number }} your favorite number?",
        model_alias=STUB_MODEL_ALIAS,
    ),
    LLMCodeColumnConfig(
        name="code_column_python",
        prompt="Generate some python about {{ valid_reference }}.",
        code_lang="python",
        model_alias=STUB_MODEL_ALIAS,
    ),
]

INVALID_COLUMNS = [
    LLMTextColumnConfig(
        name="text_no_references",
        prompt="Generate a name for the person",
        model_alias=STUB_MODEL_ALIAS,
    ),
    LLMTextColumnConfig(
        name="text_invalid_reference",
        prompt="Generate a name for the person: {{ this_column_does_not_exist }}",
        model_alias=STUB_MODEL_ALIAS,
    ),
    LLMJudgeColumnConfig(
        name="judge_no_references",
        prompt="Judge the name for the person.",
        scores=[Mock(spec=Score)],
        model_alias=STUB_MODEL_ALIAS,
    ),
    LLMJudgeColumnConfig(
        name="judge_invalid_reference",
        prompt="Judge the name for the person: {{ this_column_does_not_exist }}",
        scores=[Mock(spec=Score)],
        model_alias=STUB_MODEL_ALIAS,
    ),
    ValidationColumnConfig(
        name="code_validation_python",
        target_columns=["code_column_missing"],
        validator_type="code",
        validator_params=CodeValidatorParams(code_lang=CodeLang.SQL_ANSI),
    ),
    ValidationColumnConfig(
        name="code_validation_ansi",
        target_columns=["code_column_python"],
        validator_type="code",
        validator_params=CodeValidatorParams(code_lang=CodeLang.SQL_ANSI),
    ),
    ValidationColumnConfig(
        name="code_validation_not_code",
        target_columns=["text_no_references"],
        validator_type="code",
        validator_params=CodeValidatorParams(code_lang=CodeLang.PYTHON),
    ),
]


COLUMNS = VALID_COLUMNS + INVALID_COLUMNS
PROCESSOR_CONFIGS = [
    DropColumnsProcessorConfig(
        name="drop_columns_processor",
        column_names=["inexistent_column"],
        build_stage=BuildStage.POST_BATCH,
    )
]
ALLOWED_REFERENCE = [c.name for c in COLUMNS]


@patch("data_designer.config.utils.validation.validate_prompt_templates")
@patch("data_designer.config.utils.validation.validate_code_validation")
@patch("data_designer.config.utils.validation.validate_expression_references")
@patch("data_designer.config.utils.validation.validate_columns_not_all_dropped")
@patch("data_designer.config.utils.validation.validate_drop_columns_processor")
def test_validate_data_designer_config(
    mock_validate_columns_not_all_dropped,
    mock_validate_expression_references,
    mock_validate_code_validation,
    mock_validate_prompt_templates,
    mock_validate_drop_columns_processor,
):
    mock_validate_columns_not_all_dropped.return_value = [
        Violation(
            column="test_column",
            type=ViolationType.ALL_COLUMNS_DROPPED,
            message="test error message",
            level=ViolationLevel.ERROR,
        )
    ]
    mock_validate_expression_references.return_value = [
        Violation(
            column="test_column",
            type=ViolationType.EXPRESSION_REFERENCE_MISSING,
            message="test error message",
            level=ViolationLevel.ERROR,
        )
    ]
    mock_validate_code_validation.return_value = [
        Violation(
            column="test_column",
            type=ViolationType.CODE_COLUMN_MISSING,
            message="test error message",
            level=ViolationLevel.ERROR,
        )
    ]
    mock_validate_prompt_templates.return_value = [
        Violation(
            column="test_column",
            type=ViolationType.PROMPT_WITHOUT_REFERENCES,
            message="test error message",
            level=ViolationLevel.ERROR,
        )
    ]
    mock_validate_drop_columns_processor.return_value = [
        Violation(
            column="test_column",
            type=ViolationType.INVALID_COLUMN,
            message="test error message",
            level=ViolationLevel.ERROR,
        )
    ]
    violations = validate_data_designer_config(COLUMNS, PROCESSOR_CONFIGS, ALLOWED_REFERENCE)
    assert len(violations) == 5
    mock_validate_columns_not_all_dropped.assert_called_once()
    mock_validate_expression_references.assert_called_once()
    mock_validate_code_validation.assert_called_once()
    mock_validate_prompt_templates.assert_called_once()
    mock_validate_drop_columns_processor.assert_called_once()


def test_validate_prompt_templates():
    violations = validate_prompt_templates(COLUMNS, ALLOWED_REFERENCE)
    assert len(violations) == 4
    assert violations[0].type == ViolationType.PROMPT_WITHOUT_REFERENCES
    assert violations[1].type == ViolationType.INVALID_REFERENCE
    assert violations[2].type == ViolationType.PROMPT_WITHOUT_REFERENCES
    assert violations[3].type == ViolationType.INVALID_REFERENCE


def test_validate_code_validation():
    violations = validate_code_validation(COLUMNS)
    assert len(violations) == 3
    assert violations[0].type == ViolationType.CODE_COLUMN_MISSING
    assert violations[1].type == ViolationType.CODE_LANG_MISMATCH
    assert violations[2].type == ViolationType.CODE_COLUMN_NOT_CODE


def test_validate_detect_f_string_syntax():
    columns = VALID_COLUMNS
    columns.append(
        LLMTextColumnConfig(
            name="f_string_ref",
            prompt="Why is {random_number} your favorite number? {{ valid_reference }}",
            model_alias=STUB_MODEL_ALIAS,
        )
    )
    violations = validate_prompt_templates(columns, [c.name for c in columns])
    assert len(violations) == 1
    assert violations[0].type == ViolationType.F_STRING_SYNTAX
    assert violations[0].level == ViolationLevel.WARNING


def test_validate_column_config_with_multi_modal_context():
    column = LLMTextColumnConfig(
        name="image_description",
        prompt="Describe the image in no less that 10 sentences.",
        model_alias=STUB_MODEL_ALIAS,
        multi_modal_context=[ImageContext(column_name="image_url", data_type=ModalityDataType.URL)],
    )

    violations = validate_prompt_templates([column], [column.name])
    # there should be no violations because the prompt does not reference any columns and it's not necessary
    # when multi modal context is provided
    assert len(violations) == 0


def test_validate_columns_not_all_dropped():
    violations = validate_columns_not_all_dropped(
        [
            SamplerColumnConfig(
                name="random_number",
                sampler_type="uniform",
                params={"low": 0, "high": 10},
                drop=True,
            ),
            LLMTextColumnConfig(
                name="valid_reference",
                prompt="Why is {{ random_number }} your favorite number?",
                model_alias=STUB_MODEL_ALIAS,
                drop=True,
            ),
        ]
    )
    assert len(violations) == 1
    assert violations[0].type == ViolationType.ALL_COLUMNS_DROPPED


def test_validate_expression_references():
    violations = validate_expression_references(
        [
            ExpressionColumnConfig(
                name="expression_column",
                expr="{{ random_number }}",
                dtype="int",
            ),
        ],
        allowed_references=["some_other_column"],
    )
    assert len(violations) == 1
    assert violations[0].type == ViolationType.EXPRESSION_REFERENCE_MISSING


@patch("data_designer.config.utils.validation.Console.print")
def test_rich_print_violations(mock_console_print):
    rich_print_violations([])
    mock_console_print.assert_not_called()

    rich_print_violations(
        [
            Violation(
                column="test_column",
                type=ViolationType.EXPRESSION_REFERENCE_MISSING,
                message="test error message",
                level=ViolationLevel.ERROR,
            )
        ]
    )
    mock_console_print.assert_called_once()
