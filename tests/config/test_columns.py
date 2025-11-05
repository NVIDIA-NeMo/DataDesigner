# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import ValidationError
import pytest

from data_designer.config.columns import (
    DataDesignerColumnType,
    ExpressionColumnConfig,
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    Score,
    SeedDatasetColumnConfig,
    ValidationColumnConfig,
    get_column_config_from_kwargs,
)
from data_designer.config.errors import InvalidConfigError
from data_designer.config.sampler_params import SamplerType, UUIDSamplerParams
from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.utils.errors import UserJinjaTemplateSyntaxError
from data_designer.config.validator_params import CodeValidatorParams

stub_prompt = "test_prompt {{some_column}}"
stub_system_prompt = "test_system_prompt {{some_other_column}}"
stub_model_alias = "test_model"


def test_data_designer_column_type_get_display_order():
    assert DataDesignerColumnType.get_display_order() == [
        DataDesignerColumnType.SEED_DATASET,
        DataDesignerColumnType.SAMPLER,
        DataDesignerColumnType.LLM_TEXT,
        DataDesignerColumnType.LLM_CODE,
        DataDesignerColumnType.LLM_STRUCTURED,
        DataDesignerColumnType.LLM_JUDGE,
        DataDesignerColumnType.VALIDATION,
        DataDesignerColumnType.EXPRESSION,
        DataDesignerColumnType.CUSTOM,
    ]


def test_data_designer_column_type_has_prompt_templates():
    assert DataDesignerColumnType.LLM_TEXT.has_prompt_templates
    assert DataDesignerColumnType.LLM_CODE.has_prompt_templates
    assert DataDesignerColumnType.LLM_STRUCTURED.has_prompt_templates
    assert DataDesignerColumnType.LLM_JUDGE.has_prompt_templates
    assert not DataDesignerColumnType.SAMPLER.has_prompt_templates
    assert not DataDesignerColumnType.VALIDATION.has_prompt_templates
    assert not DataDesignerColumnType.EXPRESSION.has_prompt_templates
    assert not DataDesignerColumnType.SEED_DATASET.has_prompt_templates


def test_data_designer_column_type_is_dag_column_type():
    assert DataDesignerColumnType.EXPRESSION.is_dag_column_type
    assert DataDesignerColumnType.LLM_CODE.is_dag_column_type
    assert DataDesignerColumnType.LLM_JUDGE.is_dag_column_type
    assert DataDesignerColumnType.LLM_STRUCTURED.is_dag_column_type
    assert DataDesignerColumnType.LLM_TEXT.is_dag_column_type
    assert DataDesignerColumnType.VALIDATION.is_dag_column_type
    assert not DataDesignerColumnType.SAMPLER.is_dag_column_type
    assert not DataDesignerColumnType.SEED_DATASET.is_dag_column_type


def test_sampler_column_config():
    sampler_column_config = SamplerColumnConfig(
        name="test_sampler",
        sampler_type=SamplerType.UUID,
        params=UUIDSamplerParams(prefix="test_", short_form=True),
    )
    assert sampler_column_config.name == "test_sampler"
    assert sampler_column_config.sampler_type == SamplerType.UUID
    assert sampler_column_config.params.prefix == "test_"
    assert sampler_column_config.params.short_form is True
    assert sampler_column_config.column_type == DataDesignerColumnType.SAMPLER
    assert sampler_column_config.required_columns == []
    assert sampler_column_config.side_effect_columns == []


def test_llm_text_column_config():
    llm_text_column_config = LLMTextColumnConfig(
        name="test_llm_text",
        prompt=stub_prompt,
        model_alias=stub_model_alias,
        system_prompt=stub_system_prompt,
    )
    assert llm_text_column_config.name == "test_llm_text"
    assert llm_text_column_config.prompt == stub_prompt
    assert llm_text_column_config.model_alias == stub_model_alias
    assert llm_text_column_config.system_prompt == stub_system_prompt
    assert llm_text_column_config.column_type == DataDesignerColumnType.LLM_TEXT
    assert set(llm_text_column_config.required_columns) == {"some_column", "some_other_column"}
    assert llm_text_column_config.side_effect_columns == ["test_llm_text__reasoning_trace"]

    # invalid prompt
    with pytest.raises(
        UserJinjaTemplateSyntaxError, match="Encountered a syntax error in the provided Jinja2 template"
    ):
        LLMTextColumnConfig(
            name="test_llm_text",
            prompt="test_prompt {{some_column",
            model_alias=stub_model_alias,
            system_prompt=stub_system_prompt,
        )

    # invalid system prompt
    with pytest.raises(
        UserJinjaTemplateSyntaxError, match="Encountered a syntax error in the provided Jinja2 template"
    ):
        LLMTextColumnConfig(
            name="test_llm_text",
            prompt=stub_prompt,
            model_alias=stub_model_alias,
            system_prompt="test_system_prompt {{some_other_column",
        )


def test_llm_code_column_config():
    llm_code_column_config = LLMCodeColumnConfig(
        name="test_llm_code",
        prompt=stub_prompt,
        code_lang=CodeLang.PYTHON,
        model_alias=stub_model_alias,
    )
    assert llm_code_column_config.column_type == DataDesignerColumnType.LLM_CODE


def test_llm_structured_column_config():
    llm_structured_column_config = LLMStructuredColumnConfig(
        name="test_llm_structured",
        prompt=stub_prompt,
        output_format={"type": "object", "properties": {"some_property": {"type": "string"}}},
        model_alias=stub_model_alias,
    )
    assert llm_structured_column_config.column_type == DataDesignerColumnType.LLM_STRUCTURED
    with pytest.raises(ValidationError):
        LLMStructuredColumnConfig(
            name="test_llm_structured",
            prompt=stub_prompt,
            output_format="invalid output format",
            model_alias="test_model",
        )


def test_llm_judge_column_config():
    llm_judge_column_config = LLMJudgeColumnConfig(
        name="test_llm_judge",
        prompt=stub_prompt,
        scores=[Score(name="test_score", description="test", options={"0": "Not Good", "1": "Good"})],
        model_alias=stub_model_alias,
    )
    assert llm_judge_column_config.column_type == DataDesignerColumnType.LLM_JUDGE


def test_expression_column_config():
    expression_column_config = ExpressionColumnConfig(
        name="test_expression",
        expr="1 + 1 * {{some_column}}",
        dtype="str",
    )
    assert expression_column_config.column_type == DataDesignerColumnType.EXPRESSION
    assert expression_column_config.expr == "1 + 1 * {{some_column}}"
    assert expression_column_config.dtype == "str"
    assert expression_column_config.required_columns == ["some_column"]
    assert expression_column_config.side_effect_columns == []

    with pytest.raises(
        UserJinjaTemplateSyntaxError, match="Encountered a syntax error in the provided Jinja2 template"
    ):
        ExpressionColumnConfig(
            name="test_expression",
            expr="1 + {{some_column",
            dtype="str",
        )

    with pytest.raises(
        InvalidConfigError, match="Expression column 'test_expression' has an empty or whitespace-only expression"
    ):
        ExpressionColumnConfig(
            name="test_expression",
            expr="",
            dtype="str",
        )


def test_validation_column_config():
    validation_column_config = ValidationColumnConfig(
        name="test_validation",
        target_columns=["test_column"],
        validator_type="code",
        validator_params=CodeValidatorParams(code_lang=CodeLang.PYTHON),
        batch_size=5,
    )
    assert validation_column_config.column_type == DataDesignerColumnType.VALIDATION
    assert validation_column_config.target_columns == ["test_column"]
    assert validation_column_config.required_columns == ["test_column"]
    assert validation_column_config.side_effect_columns == []
    assert validation_column_config.batch_size == 5


def test_get_column_config_from_kwargs():
    assert isinstance(
        get_column_config_from_kwargs(
            name="test_llm_text",
            column_type=DataDesignerColumnType.LLM_TEXT,
            prompt=stub_prompt,
            model_alias=stub_model_alias,
            system_prompt=stub_system_prompt,
        ),
        LLMTextColumnConfig,
    )

    assert isinstance(
        get_column_config_from_kwargs(
            name="test_llm_code",
            column_type=DataDesignerColumnType.LLM_CODE,
            prompt=stub_prompt,
            code_lang=CodeLang.PYTHON,
            model_alias=stub_model_alias,
        ),
        LLMCodeColumnConfig,
    )

    assert isinstance(
        get_column_config_from_kwargs(
            name="test_llm_structured",
            column_type=DataDesignerColumnType.LLM_STRUCTURED,
            prompt=stub_prompt,
            output_format={"type": "object", "properties": {"some_property": {"type": "string"}}},
            model_alias=stub_model_alias,
        ),
        LLMStructuredColumnConfig,
    )

    assert isinstance(
        get_column_config_from_kwargs(
            name="test_llm_judge",
            column_type=DataDesignerColumnType.LLM_JUDGE,
            prompt=stub_prompt,
            scores=[Score(name="test_score", description="test", options={"0": "Not Good", "1": "Good"})],
            model_alias=stub_model_alias,
        ),
        LLMJudgeColumnConfig,
    )

    assert isinstance(
        get_column_config_from_kwargs(
            name="test_validation",
            column_type=DataDesignerColumnType.VALIDATION,
            target_columns=["test_column"],
            validator_type="code",
            validator_params=CodeValidatorParams(code_lang=CodeLang.PYTHON),
        ),
        ValidationColumnConfig,
    )

    assert isinstance(
        get_column_config_from_kwargs(
            name="test_expression",
            column_type=DataDesignerColumnType.EXPRESSION,
            expr="1 + 1 * {{some_column}}",
            dtype="str",
        ),
        ExpressionColumnConfig,
    )

    # sampler params is a dictionary
    assert isinstance(
        get_column_config_from_kwargs(
            name="test_sampler",
            column_type=DataDesignerColumnType.SAMPLER,
            sampler_type=SamplerType.UUID,
            params=dict(prefix="test_", short_form=True),
        ),
        SamplerColumnConfig,
    )

    # sampler params is a concrete object
    assert isinstance(
        get_column_config_from_kwargs(
            name="test_sampler",
            column_type=DataDesignerColumnType.SAMPLER,
            sampler_type=SamplerType.UUID,
            params=UUIDSamplerParams(prefix="test_", short_form=True),
        ),
        SamplerColumnConfig,
    )

    # sampler params is invalid
    with pytest.raises(
        InvalidConfigError,
        match="Invalid params for sampler column 'test_sampler'. Expected a dictionary or an instance",
    ):
        assert isinstance(
            get_column_config_from_kwargs(
                name="test_sampler",
                column_type=DataDesignerColumnType.SAMPLER,
                sampler_type=SamplerType.UUID,
                params="invalid params",
            ),
            SamplerColumnConfig,
        )

    # sampler type is missing
    with pytest.raises(InvalidConfigError, match="`sampler_type` is required for sampler column 'test_sampler'."):
        assert isinstance(
            get_column_config_from_kwargs(
                name="test_sampler",
                column_type=DataDesignerColumnType.SAMPLER,
            ),
            SamplerColumnConfig,
        )

    assert isinstance(
        get_column_config_from_kwargs(
            name="test_seed_dataset",
            column_type=DataDesignerColumnType.SEED_DATASET,
        ),
        SeedDatasetColumnConfig,
    )
