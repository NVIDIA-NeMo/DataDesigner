# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import ValidationError
import pytest

from data_designer.config.column_configs import (
    ExpressionColumnConfig,
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    Score,
    SeedDatasetColumnConfig,
    ValidationColumnConfig,
)
from data_designer.config.column_types import (
    DataDesignerColumnType,
    column_type_is_llm_generated,
    column_type_used_in_execution_dag,
    get_column_config_from_kwargs,
    get_column_display_order,
)
from data_designer.config.errors import InvalidConfigError
from data_designer.config.sampler_params import (
    CategorySamplerParams,
    GaussianSamplerParams,
    PersonFromFakerSamplerParams,
    PersonSamplerParams,
    SamplerType,
    UniformSamplerParams,
    UUIDSamplerParams,
)
from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.utils.errors import UserJinjaTemplateSyntaxError
from data_designer.config.validator_params import CodeValidatorParams

stub_prompt = "test_prompt {{some_column}}"
stub_system_prompt = "test_system_prompt {{some_other_column}}"
stub_model_alias = "test_model"


def test_data_designer_column_type_get_display_order():
    assert get_column_display_order() == [
        DataDesignerColumnType.SEED_DATASET,
        DataDesignerColumnType.SAMPLER,
        DataDesignerColumnType.LLM_TEXT,
        DataDesignerColumnType.LLM_CODE,
        DataDesignerColumnType.LLM_STRUCTURED,
        DataDesignerColumnType.LLM_JUDGE,
        DataDesignerColumnType.EMBEDDING,
        DataDesignerColumnType.IMAGE_GENERATION,
        DataDesignerColumnType.VALIDATION,
        DataDesignerColumnType.EXPRESSION,
    ]


def test_data_designer_column_type_is_llm_generated():
    assert column_type_is_llm_generated(DataDesignerColumnType.LLM_TEXT)
    assert column_type_is_llm_generated(DataDesignerColumnType.LLM_CODE)
    assert column_type_is_llm_generated(DataDesignerColumnType.LLM_STRUCTURED)
    assert column_type_is_llm_generated(DataDesignerColumnType.LLM_JUDGE)
    assert column_type_is_llm_generated(DataDesignerColumnType.EMBEDDING)
    assert column_type_is_llm_generated(DataDesignerColumnType.IMAGE_GENERATION)
    assert not column_type_is_llm_generated(DataDesignerColumnType.SAMPLER)
    assert not column_type_is_llm_generated(DataDesignerColumnType.VALIDATION)
    assert not column_type_is_llm_generated(DataDesignerColumnType.EXPRESSION)
    assert not column_type_is_llm_generated(DataDesignerColumnType.SEED_DATASET)


def test_data_designer_column_type_is_in_dag():
    assert column_type_used_in_execution_dag(DataDesignerColumnType.EXPRESSION)
    assert column_type_used_in_execution_dag(DataDesignerColumnType.LLM_CODE)
    assert column_type_used_in_execution_dag(DataDesignerColumnType.LLM_JUDGE)
    assert column_type_used_in_execution_dag(DataDesignerColumnType.LLM_STRUCTURED)
    assert column_type_used_in_execution_dag(DataDesignerColumnType.LLM_TEXT)
    assert column_type_used_in_execution_dag(DataDesignerColumnType.VALIDATION)
    assert column_type_used_in_execution_dag(DataDesignerColumnType.EMBEDDING)
    assert column_type_used_in_execution_dag(DataDesignerColumnType.IMAGE_GENERATION)
    assert not column_type_used_in_execution_dag(DataDesignerColumnType.SAMPLER)
    assert not column_type_used_in_execution_dag(DataDesignerColumnType.SEED_DATASET)


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


def test_sampler_column_config_discriminated_union_with_dict_params():
    """Test that sampler_type field is automatically injected into params dict."""
    config = SamplerColumnConfig(
        name="test_uniform",
        sampler_type=SamplerType.UNIFORM,
        params={"low": 0.0, "high": 1.0, "decimal_places": 2},
    )
    assert config.name == "test_uniform"
    assert config.sampler_type == SamplerType.UNIFORM
    assert isinstance(config.params, UniformSamplerParams)
    assert config.params.sampler_type == SamplerType.UNIFORM
    assert config.params.low == 0.0
    assert config.params.high == 1.0
    assert config.params.decimal_places == 2


def test_sampler_column_config_discriminated_union_with_explicit_sampler_type():
    """Test that explicit sampler_type in params dict is preserved."""
    config = SamplerColumnConfig(
        name="test_category",
        sampler_type=SamplerType.CATEGORY,
        params={"sampler_type": "category", "values": ["A", "B", "C"], "weights": [0.5, 0.3, 0.2]},
    )
    assert config.name == "test_category"
    assert config.sampler_type == SamplerType.CATEGORY
    assert isinstance(config.params, CategorySamplerParams)
    assert config.params.sampler_type == SamplerType.CATEGORY
    assert config.params.values == ["A", "B", "C"]


def test_sampler_column_config_discriminated_union_serialization():
    """Test that discriminated union works correctly with serialization/deserialization."""
    config = SamplerColumnConfig(
        name="test_person",
        sampler_type=SamplerType.PERSON,
        params={"locale": "en_US", "sex": "Female", "age_range": [25, 45]},
    )

    # Serialize
    serialized = config.model_dump()
    assert "sampler_type" in serialized["params"]
    assert serialized["params"]["sampler_type"] == "person"

    # Deserialize
    deserialized = SamplerColumnConfig(**serialized)
    assert isinstance(deserialized.params, PersonSamplerParams)
    assert deserialized.params.locale == "en_US"
    assert deserialized.params.sex == "Female"
    assert deserialized.params.age_range == [25, 45]


def test_sampler_column_config_discriminated_union_person_vs_person_from_faker():
    """Test that discriminated union correctly distinguishes between person and person_from_faker."""
    # Test person sampler (managed datasets)
    person_config = SamplerColumnConfig(
        name="test_person",
        sampler_type=SamplerType.PERSON,
        params={"locale": "en_US", "sex": "Male", "age_range": [30, 50]},
    )
    assert isinstance(person_config.params, PersonSamplerParams)
    assert person_config.params.sampler_type == SamplerType.PERSON
    assert person_config.params.locale == "en_US"

    # Test person_from_faker sampler (Faker-based)
    person_faker_config = SamplerColumnConfig(
        name="test_person_faker",
        sampler_type=SamplerType.PERSON_FROM_FAKER,
        params={"locale": "en_GB", "sex": "Female", "age_range": [20, 40]},
    )
    assert isinstance(person_faker_config.params, PersonFromFakerSamplerParams)
    assert person_faker_config.params.sampler_type == SamplerType.PERSON_FROM_FAKER
    assert person_faker_config.params.locale == "en_GB"

    # Verify they are different types
    assert type(person_config.params) is not type(person_faker_config.params)
    assert isinstance(person_config.params, PersonSamplerParams)
    assert isinstance(person_faker_config.params, PersonFromFakerSamplerParams)


def test_sampler_column_config_discriminated_union_with_conditional_params():
    """Test that sampler_type is injected into conditional_params as well."""
    config = SamplerColumnConfig(
        name="test_gaussian",
        sampler_type=SamplerType.GAUSSIAN,
        params={"mean": 0.0, "stddev": 1.0},
        conditional_params={"age > 21": {"mean": 5.0, "stddev": 2.0}},
    )

    assert isinstance(config.params, GaussianSamplerParams)
    assert config.params.mean == 0.0
    assert config.params.stddev == 1.0

    # Check conditional params
    assert "age > 21" in config.conditional_params
    cond_param = config.conditional_params["age > 21"]
    assert isinstance(cond_param, GaussianSamplerParams)
    assert cond_param.sampler_type == SamplerType.GAUSSIAN
    assert cond_param.mean == 5.0
    assert cond_param.stddev == 2.0


def test_sampler_column_config_discriminated_union_wrong_params_type():
    """Test that discriminated union rejects params that don't match the sampler_type."""
    with pytest.raises(ValidationError):
        SamplerColumnConfig(
            name="test_wrong_params",
            sampler_type=SamplerType.UNIFORM,
            params={"values": ["A", "B"]},  # Category params for uniform sampler
        )
