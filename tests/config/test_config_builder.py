# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from pydantic import BaseModel, ValidationError
import pytest
import yaml

from data_designer.config.analysis.column_profilers import JudgeScoreProfilerConfig
from data_designer.config.column_configs import (
    ExpressionColumnConfig,
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    Score,
    ValidationColumnConfig,
)
from data_designer.config.column_types import DataDesignerColumnType, get_column_config_from_kwargs
from data_designer.config.config_builder import BuilderConfig, DataDesignerConfigBuilder
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.datastore import DatastoreSettings
from data_designer.config.errors import BuilderConfigurationError, InvalidColumnTypeError, InvalidConfigError
from data_designer.config.models import InferenceParameters, ModelConfig
from data_designer.config.sampler_constraints import ColumnInequalityConstraint, ScalarInequalityConstraint
from data_designer.config.sampler_params import SamplerType, UUIDSamplerParams
from data_designer.config.seed import DatastoreSeedDatasetReference, SamplingStrategy
from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.utils.info import ConfigBuilderInfo
from data_designer.config.validator_params import CodeValidatorParams


class DummyStructuredModel(BaseModel):
    stub: str


@pytest.fixture
def mock_fetch_seed_dataset_column_names():
    with patch("data_designer.config.config_builder.fetch_seed_dataset_column_names") as mock_fetch_seed:
        mock_fetch_seed.return_value = ["id", "name", "age", "city"]
        yield mock_fetch_seed


@pytest.fixture
def stub_data_designer_builder(stub_data_designer_builder_config_str):
    with patch("data_designer.config.config_builder.fetch_seed_dataset_column_names") as mock_fetch_seed:
        mock_fetch_seed.return_value = ["id", "name", "age", "city"]
        yield DataDesignerConfigBuilder.from_config(config=stub_data_designer_builder_config_str)


def test_loading_model_configs_in_constructor(stub_model_configs):
    stub_model_configs_dict = [mc.model_dump() for mc in stub_model_configs]
    # test loading model configs from a list
    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    assert builder.model_configs == stub_model_configs

    # test loading model configs from a yaml file
    with tempfile.NamedTemporaryFile(prefix="model_configs", suffix=".yaml") as tmp_file:
        model_configs = {"model_configs": stub_model_configs_dict}
        tmp_file.write(yaml.safe_dump(model_configs).encode("utf-8"))
        tmp_file.flush()
        builder = DataDesignerConfigBuilder(model_configs=tmp_file.name)
        assert builder.model_configs == stub_model_configs

    # test failure when the list is not grouped under model_configs in the yaml file
    with tempfile.NamedTemporaryFile(prefix="model_configs", suffix=".yaml") as tmp_file:
        model_configs = {"invalid": stub_model_configs_dict}
        tmp_file.write(yaml.safe_dump(model_configs).encode("utf-8"))
        tmp_file.flush()
        with pytest.raises(InvalidConfigError):
            builder = DataDesignerConfigBuilder(model_configs=tmp_file.name)


def test_from_config(stub_data_designer_builder_config_str, mock_fetch_seed_dataset_column_names):
    builder = DataDesignerConfigBuilder.from_config(config=stub_data_designer_builder_config_str)

    assert isinstance(builder.get_column_config(name="code_id"), SamplerColumnConfig)
    assert isinstance(builder.get_column_config(name="text"), LLMTextColumnConfig)
    assert isinstance(builder.get_column_config(name="code"), LLMCodeColumnConfig)
    assert isinstance(builder.get_column_config(name="code_validation_result"), ValidationColumnConfig)
    assert isinstance(builder.get_column_config(name="code_judge_result"), LLMJudgeColumnConfig)
    assert builder.build().model_configs[0].alias == "my_own_code_model"

    # test construction from a concrete config object
    config_dict = yaml.safe_load(stub_data_designer_builder_config_str)
    builder_config_object = BuilderConfig.model_validate(config_dict)
    builder_from_object = DataDesignerConfigBuilder.from_config(
        config=builder_config_object,
    )
    assert isinstance(builder_from_object.get_column_config(name="code_id"), SamplerColumnConfig)


def test_info(stub_data_designer_builder):
    assert stub_data_designer_builder.info is not None
    assert isinstance(stub_data_designer_builder.info, ConfigBuilderInfo)


def test_add_column_with_types(stub_empty_builder):
    stub_empty_builder.add_column(
        SamplerColumnConfig(
            name="test_id",
            sampler_type=SamplerType.UUID,
            params=UUIDSamplerParams(prefix="code_", short_form=True, uppercase=True),
        )
    )
    stub_empty_builder.add_column(
        LLMCodeColumnConfig(
            name="test_code",
            prompt="Write some zig but call it Python.",
            code_lang=CodeLang.PYTHON,
            model_alias="stub-code-alias",
        )
    )
    stub_empty_builder.add_column(
        LLMStructuredColumnConfig(
            name="test_structured_output",
            prompt="Generate a structured output",
            output_format=DummyStructuredModel,
            model_alias="stub-structured-alias",
        )
    )
    stub_empty_builder.add_column(
        LLMJudgeColumnConfig(
            name="test_judge",
            prompt="Judge this",
            scores=[
                Score(
                    name="test_rubric",
                    description="test",
                    options={"0": "Not Good", "1": "Good"},
                )
            ],
            model_alias="stub-judge-alias",
        )
    )
    stub_empty_builder.add_column(
        ValidationColumnConfig(
            name="test_validation",
            target_columns=["test_code"],
            validator_type="code",
            validator_params=CodeValidatorParams(code_lang=CodeLang.PYTHON),
        )
    )
    stub_empty_builder.add_column(
        ExpressionColumnConfig(
            name="test_expression",
            expr="1 + 1",
        )
    )
    assert stub_empty_builder.get_column_config("test_id").name == "test_id"
    assert stub_empty_builder.get_column_config("test_code").name == "test_code"
    assert stub_empty_builder.get_column_config("test_structured_output").name == "test_structured_output"
    assert (
        stub_empty_builder.get_column_config("test_structured_output").output_format
        == DummyStructuredModel.model_json_schema()
    )
    assert stub_empty_builder.get_column_config("test_judge").name == "test_judge"
    assert stub_empty_builder.get_column_config("test_validation").name == "test_validation"
    assert len(stub_empty_builder.get_column_configs()) == 6

    # replace existing column
    stub_empty_builder.add_column(
        LLMCodeColumnConfig(
            name="test_code",
            prompt="Write some zig but call it Swift.",
            code_lang=CodeLang.SWIFT,
            model_alias="stub-code-alias",
        )
    )
    assert stub_empty_builder.get_column_config(name="test_code").prompt == "Write some zig but call it Swift."


def test_add_column_with_kwargs(stub_empty_builder):
    stub_empty_builder.add_column(
        name="test_id",
        column_type="sampler",
        sampler_type=SamplerType.UUID,
        params={"prefix": "code_", "short_form": True, "uppercase": True},
    )
    stub_empty_builder.add_column(
        name="test_code",
        column_type="llm-code",
        prompt="Write some zig but call it Python.",
        code_lang=CodeLang.PYTHON,
        model_alias="stub-code-alias",
    )
    stub_empty_builder.add_column(
        name="test_structured_output",
        column_type="llm-structured",
        model_alias="stub-structured-alias",
        prompt="Generate a structured output",
        output_format=DummyStructuredModel,
    )
    stub_empty_builder.add_column(
        name="test_judge",
        column_type="llm-judge",
        model_alias="stub-judge-alias",
        prompt="Judge this",
        scores=[
            {
                "name": "test_rubric",
                "description": "test",
                "options": {"0": "Not Good", "1": "Good"},
            }
        ],
    )
    stub_empty_builder.add_column(
        name="test_validation",
        column_type="validation",
        target_columns=["test_code"],
        validator_type="code",
        validator_params=CodeValidatorParams(code_lang=CodeLang.PYTHON),
    )
    stub_empty_builder.add_column(
        name="test_expression",
        column_type="expression",
        expr="1 + 1",
    )
    assert stub_empty_builder.get_column_config("test_id").name == "test_id"
    assert stub_empty_builder.get_column_config("test_code").name == "test_code"
    assert stub_empty_builder.get_column_config("test_structured_output").name == "test_structured_output"
    assert (
        stub_empty_builder.get_column_config("test_structured_output").output_format
        == DummyStructuredModel.model_json_schema()
    )
    assert stub_empty_builder.get_column_config("test_judge").name == "test_judge"
    assert stub_empty_builder.get_column_config("test_validation").name == "test_validation"
    assert len(stub_empty_builder.get_column_configs()) == 6

    # replace existing column
    stub_empty_builder.add_column(
        name="test_code",
        column_type="llm-code",
        prompt="Write some zig but call it Swift.",
        code_lang=CodeLang.SWIFT,
        model_alias="stub-code-alias",
    )
    assert stub_empty_builder.get_column_config(name="test_code").prompt == "Write some zig but call it Swift."


def test_add_column_exceptions(stub_empty_builder):
    with pytest.raises(
        BuilderConfigurationError,
        match="You must provide either a 'column_config' object or 'name' \\*and\\* 'column_type'",
    ):
        stub_empty_builder.add_column(
            prompt="Write some zig but call it Swift.",
        )
    with patch("data_designer.config.config_builder.get_column_config_from_kwargs") as mock_get_column_config:
        mock_get_column_config.return_value = "invalid"
        with pytest.raises(InvalidColumnTypeError, match="Invalid column config object: 'invalid'."):
            stub_empty_builder.add_column(
                name="test_code",
                column_type="llm-code",
                code_lang=CodeLang.SWIFT,
                model_alias="stub-code-alias",
                prompt="Test",
            )


def test_add_constraint_with_types(stub_empty_builder):
    stub_empty_builder.add_column(
        name="no_constraint",
        column_type=DataDesignerColumnType.SAMPLER,
        sampler_type=SamplerType.UNIFORM,
        params={"low": 1, "high": 100},
    )
    stub_empty_builder.add_column(
        name="age",
        column_type=DataDesignerColumnType.SAMPLER,
        sampler_type=SamplerType.GAUSSIAN,
        params={"mean": 35, "stddev": 5},
    )
    stub_empty_builder.add_constraint(ScalarInequalityConstraint(target_column="age", operator="gt", rhs=30))
    stub_empty_builder.add_constraint(ScalarInequalityConstraint(target_column="age", operator="lt", rhs=35))
    stub_empty_builder.add_constraint(ColumnInequalityConstraint(target_column="height", operator="gt", rhs="age"))

    assert len(stub_empty_builder.get_constraints(target_column="no_constraint")) == 0
    assert len(stub_empty_builder.get_constraints(target_column="age")) == 2
    assert len(stub_empty_builder.get_constraints(target_column="height")) == 1
    assert stub_empty_builder.get_constraints(target_column="age")[0].operator == "gt"
    assert stub_empty_builder.get_constraints(target_column="age")[1].rhs == 35

    stub_empty_builder.delete_constraints(target_column="height")
    assert len(stub_empty_builder.get_constraints(target_column="height")) == 0


def test_add_constraint_with_kwargs(stub_empty_builder):
    stub_empty_builder.add_column(
        name="age",
        column_type=DataDesignerColumnType.SAMPLER,
        sampler_type=SamplerType.UNIFORM,
        params={"low": 1, "high": 100},
    )
    stub_empty_builder.add_column(
        name="height",
        column_type=DataDesignerColumnType.SAMPLER,
        sampler_type=SamplerType.UNIFORM,
        params={"low": 15, "high": 200},
    )
    stub_empty_builder.add_constraint(constraint_type="scalar_inequality", target_column="age", operator="gt", rhs=30)
    stub_empty_builder.add_constraint(
        constraint_type="column_inequality", target_column="age", operator="gt", rhs="height"
    )

    assert len(stub_empty_builder.get_constraints(target_column="age")) == 2
    assert stub_empty_builder.get_constraints(target_column="age")[0].operator == "gt"
    assert stub_empty_builder.get_constraints(target_column="age")[0].rhs == 30
    assert stub_empty_builder.get_constraints(target_column="age")[1].rhs == "height"
    assert stub_empty_builder.get_constraints(target_column="age")[1].operator == "gt"

    with pytest.raises(BuilderConfigurationError):
        stub_empty_builder.add_constraint(constraint_type="invalid", target_column="age", operator="gt", rhs=30)

    with pytest.raises(ValidationError):
        stub_empty_builder.add_constraint(
            constraint_type="scalar_inequality", target_column="age", operator="gt", rhs="string"
        )


def test_add_constraint_exceptions(stub_empty_builder):
    with pytest.raises(
        BuilderConfigurationError,
        match="You must provide either a 'constraint' object or 'constraint_type' with additional keyword arguments.",
    ):
        stub_empty_builder.add_constraint()
    with pytest.raises(
        BuilderConfigurationError,
        match="Invalid constraint type: invalid. Valid options are: scalar_inequality, column_inequality",
    ):
        stub_empty_builder.add_constraint(constraint_type="invalid", target_column="age", operator="gt", rhs=30)
    with pytest.raises(
        BuilderConfigurationError,
        match="Invalid constraint object. Valid constraint options are: ScalarInequalityConstraint, ColumnInequalityConstraint",
    ):
        stub_empty_builder.add_constraint(constraint="invalid")
    with pytest.raises(ValidationError):
        stub_empty_builder.add_constraint(
            constraint_type="scalar_inequality", target_column="age", operator="gt", rhs="string"
        )


def test_add_profiler(stub_empty_builder):
    judge_profiler_config = JudgeScoreProfilerConfig(model_alias="test-alias", summary_score_sample_size=100)
    stub_empty_builder.add_profiler(judge_profiler_config)
    assert len(stub_empty_builder.get_profilers()) == 1
    assert stub_empty_builder.get_profilers()[0] == judge_profiler_config

    with pytest.raises(
        BuilderConfigurationError, match="Invalid profiler object. Valid profiler options are: JudgeScoreProfilerConfig"
    ):
        stub_empty_builder.add_profiler("invalid")


def test_build(stub_data_designer_builder):
    # verify transformation to config object
    ndd_config = stub_data_designer_builder.build()
    assert isinstance(ndd_config, DataDesignerConfig)

    with patch("data_designer.config.config_builder.DataDesignerConfigBuilder.validate") as mock_validate:
        stub_data_designer_builder.build(skip_validation=True)
        mock_validate.assert_not_called()


def test_config_export_to_files(stub_data_designer_builder):
    """Test config export to JSON and YAML files via DataDesignerConfig methods."""
    ndd_config = stub_data_designer_builder.build()

    # verify transformation to dict
    ndd_config_dict = ndd_config.to_dict()
    assert isinstance(ndd_config_dict, dict)

    # verify config export to JSON file
    with tempfile.NamedTemporaryFile(prefix="config", suffix=".json") as tmp_file:
        ndd_config.to_json(path=tmp_file.name)
        with open(tmp_file.name, "r") as f:
            assert json.loads(f.read())["model_configs"] == ndd_config.to_dict()["model_configs"]

    # verify config export to YAML file
    with tempfile.NamedTemporaryFile(prefix="config", suffix=".yaml") as tmp_file:
        ndd_config.to_yaml(path=tmp_file.name)
        with open(tmp_file.name, "r") as f:
            deserialized_config = yaml.safe_load(f.read())
            assert deserialized_config["model_configs"] == ndd_config.to_dict()["model_configs"]
            # verify enums are rendered as plain strings in the yaml file
            assert isinstance(deserialized_config["model_configs"][0], dict)

    # verify config export to .yml file
    with tempfile.NamedTemporaryFile(prefix="config", suffix=".yml") as tmp_file:
        ndd_config.to_yaml(path=tmp_file.name)
        with open(tmp_file.name, "r") as f:
            assert yaml.safe_load(f.read())["model_configs"] == ndd_config.to_dict()["model_configs"]


def test_delete_constraints(stub_data_designer_builder):
    assert len(stub_data_designer_builder.get_constraints(target_column="age")) == 1
    stub_data_designer_builder.delete_constraints(target_column="age")
    assert len(stub_data_designer_builder.get_constraints(target_column="age")) == 0


def test_delete_column(stub_data_designer_builder):
    assert len(stub_data_designer_builder.get_columns_of_type(DataDesignerColumnType.SAMPLER)) == 3
    stub_data_designer_builder.delete_column(column_name="code_id")
    assert len(stub_data_designer_builder.get_columns_of_type(DataDesignerColumnType.SAMPLER)) == 2

    with pytest.raises(
        BuilderConfigurationError, match="Seed columns cannot be deleted. Please update the seed dataset instead."
    ):
        stub_data_designer_builder.delete_column(column_name="age")


def test_getters(stub_data_designer_builder):
    assert len(stub_data_designer_builder.get_column_configs()) == 11
    assert stub_data_designer_builder.get_column_config(name="code_id").name == "code_id"
    assert len(stub_data_designer_builder.get_constraints(target_column="age")) == 1
    assert len(stub_data_designer_builder.get_llm_gen_columns()) == 3
    assert len(stub_data_designer_builder.get_columns_of_type(DataDesignerColumnType.SAMPLER)) == 3
    assert len(stub_data_designer_builder.get_columns_excluding_type(DataDesignerColumnType.SAMPLER)) == 8
    assert stub_data_designer_builder.get_seed_config().dataset == "test-repo/testing/data.csv"
    assert stub_data_designer_builder.num_columns_of_type(DataDesignerColumnType.SAMPLER) == 3


def test_write_config(stub_data_designer_builder):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test.yaml"
        stub_data_designer_builder.write_config(temp_path)
        assert temp_path.exists()
        assert temp_path.read_text() == stub_data_designer_builder.get_builder_config().to_yaml()

        temp_path = Path(temp_dir) / "test.yml"
        stub_data_designer_builder.write_config(temp_path)
        assert temp_path.exists()
        assert temp_path.read_text() == stub_data_designer_builder.get_builder_config().to_yaml()

        temp_path = Path(temp_dir) / "test.json"
        stub_data_designer_builder.write_config(temp_path)
        assert temp_path.exists()
        assert temp_path.read_text() == stub_data_designer_builder.get_builder_config().to_json()

        with pytest.raises(BuilderConfigurationError, match="Unsupported file type"):
            stub_data_designer_builder.write_config(temp_path.with_suffix(".txt"))


def test_validate(stub_empty_builder):
    try:
        stub_empty_builder.validate(raise_exceptions=False)
    except Exception:
        pytest.fail("Validate should not raise an exception if raise_exceptions is False")

    with pytest.raises(
        InvalidConfigError,
        match="Your configuration contains validation errors. Please address the indicated issues and try again.",
    ):
        stub_empty_builder.validate(raise_exceptions=True)

    with patch("data_designer.config.config_builder.logger") as mock_logger:
        stub_empty_builder.add_column(
            name="test_column",
            column_type=DataDesignerColumnType.SAMPLER,
            sampler_type=SamplerType.UUID,
            params={"prefix": "test_", "short_form": True, "uppercase": True},
        )
        try:
            stub_empty_builder.validate(raise_exceptions=True)
        except Exception:
            pytest.fail("Validate should not raise an exception for valid configuration")
        mock_logger.info.assert_called_once_with("✅ Validation passed")


def test_get_column_config_from_kwargs():
    # Test column creation and serialization

    llm_text_column = get_column_config_from_kwargs(
        name="test_llm_text",
        column_type=DataDesignerColumnType.LLM_TEXT,
        prompt="Write some text",
        model_alias="stub-alias",
    )
    assert isinstance(llm_text_column, LLMTextColumnConfig)
    assert llm_text_column.name == "test_llm_text"
    assert llm_text_column.prompt == "Write some text"
    assert llm_text_column.model_alias == "stub-alias"

    llm_code_column = get_column_config_from_kwargs(
        name="test_llm_code",
        column_type=DataDesignerColumnType.LLM_CODE,
        prompt="Write some code",
        code_lang="python",
        model_alias="stub-alias",
    )
    assert isinstance(llm_code_column, LLMCodeColumnConfig)
    assert llm_code_column.name == "test_llm_code"
    assert llm_code_column.prompt == "Write some code"
    assert llm_code_column.code_lang == CodeLang.PYTHON
    assert llm_code_column.model_alias == "stub-alias"

    llm_structured_column = get_column_config_from_kwargs(
        name="test_llm_structured",
        column_type=DataDesignerColumnType.LLM_STRUCTURED,
        model_alias="stub-alias",
        prompt="Generate a structured output",
        output_format=DummyStructuredModel.model_json_schema(),
    )
    assert isinstance(llm_structured_column, LLMStructuredColumnConfig)
    assert llm_structured_column.name == "test_llm_structured"
    assert llm_structured_column.prompt == "Generate a structured output"
    assert llm_structured_column.output_format == DummyStructuredModel.model_json_schema()
    assert llm_structured_column.model_alias == "stub-alias"

    llm_judge_column = get_column_config_from_kwargs(
        name="test_judge",
        column_type=DataDesignerColumnType.LLM_JUDGE,
        model_alias="stub-judge-alias",
        prompt="Judge this code",
        scores=[
            Score(
                name="test_rubric",
                description="test",
                options={"0": "Bad", "1": "Good"},
            )
        ],
    )
    assert isinstance(llm_judge_column, LLMJudgeColumnConfig)
    assert llm_judge_column.name == "test_judge"
    assert llm_judge_column.prompt == "Judge this code"
    assert len(llm_judge_column.scores) == 1

    code_validation_column = get_column_config_from_kwargs(
        name="test_validation",
        column_type=DataDesignerColumnType.VALIDATION,
        target_columns=["test_code"],
        validator_type="code",
        validator_params=CodeValidatorParams(code_lang=CodeLang.PYTHON),
    )
    assert isinstance(code_validation_column, ValidationColumnConfig)
    assert code_validation_column.name == "test_validation"
    assert code_validation_column.validator_params.code_lang == CodeLang.PYTHON
    assert code_validation_column.target_columns == ["test_code"]

    expression_column = get_column_config_from_kwargs(
        name="test_expression", column_type=DataDesignerColumnType.EXPRESSION, expr="1 + 1"
    )
    assert isinstance(expression_column, ExpressionColumnConfig)
    assert expression_column.name == "test_expression"
    assert expression_column.expr == "1 + 1"

    # Test empty expression validation
    with pytest.raises(
        InvalidConfigError, match="Expression column 'test_empty_expression' has an empty or whitespace-only expression"
    ):
        get_column_config_from_kwargs(
            name="test_empty_expression", column_type=DataDesignerColumnType.EXPRESSION, expr=""
        )

    # Test whitespace-only expression validation
    with pytest.raises(
        InvalidConfigError,
        match="Expression column 'test_whitespace_expression' has an empty or whitespace-only expression",
    ):
        get_column_config_from_kwargs(
            name="test_whitespace_expression", column_type=DataDesignerColumnType.EXPRESSION, expr="   \t\n  "
        )

    # Test Sampler columns with nullable params
    # UUID type with params provided
    sampler_column = get_column_config_from_kwargs(
        name="test_sampler",
        column_type=DataDesignerColumnType.SAMPLER,
        sampler_type=SamplerType.UUID,
        params={"prefix": "test_", "short_form": True},
    )
    assert isinstance(sampler_column, SamplerColumnConfig)
    assert sampler_column.name == "test_sampler"
    assert sampler_column.sampler_type == SamplerType.UUID
    assert sampler_column.params.prefix == "test_"
    assert sampler_column.params.short_form is True
    assert sampler_column.params.uppercase is False

    # UUID type without params provided
    sampler_column_no_params = get_column_config_from_kwargs(
        name="test_sampler_no_params",
        column_type=DataDesignerColumnType.SAMPLER,
        sampler_type=SamplerType.UUID,
    )
    assert isinstance(sampler_column_no_params, SamplerColumnConfig)
    assert sampler_column_no_params.name == "test_sampler_no_params"
    assert sampler_column_no_params.sampler_type == SamplerType.UUID
    assert sampler_column_no_params.params.prefix is None
    assert sampler_column_no_params.params.short_form is False
    assert sampler_column_no_params.params.uppercase is False

    # PERSON type with params provided (must use locale with managed dataset)
    person_sampler_column = get_column_config_from_kwargs(
        name="test_person_sampler",
        column_type=DataDesignerColumnType.SAMPLER,
        sampler_type=SamplerType.PERSON,
        params={
            "locale": "en_US",
            "sex": "Male",
            "city": "New York",
            "age_range": [18, 30],
        },
    )
    assert isinstance(person_sampler_column, SamplerColumnConfig)
    assert person_sampler_column.name == "test_person_sampler"
    assert person_sampler_column.sampler_type == SamplerType.PERSON
    assert person_sampler_column.params.locale == "en_US"
    assert person_sampler_column.params.sex == "Male"
    assert person_sampler_column.params.city == "New York"

    person_sampler_column_no_params = get_column_config_from_kwargs(
        name="test_person_sampler_no_params",
        column_type=DataDesignerColumnType.SAMPLER,
        sampler_type=SamplerType.PERSON,
    )
    assert isinstance(person_sampler_column_no_params, SamplerColumnConfig)
    assert person_sampler_column_no_params.name == "test_person_sampler_no_params"
    assert person_sampler_column_no_params.sampler_type == SamplerType.PERSON
    assert person_sampler_column_no_params.params.locale == "en_US"
    assert person_sampler_column_no_params.params.sex is None
    assert person_sampler_column_no_params.params.city is None


def test_seed_config(stub_complete_builder):
    seed_config = stub_complete_builder.get_seed_config()
    assert seed_config is not None
    assert seed_config.dataset == "test-repo/testing/data.csv"
    assert seed_config.sampling_strategy == SamplingStrategy.SHUFFLE


def test_with_seed_dataset_basic(stub_empty_builder, mock_fetch_seed_dataset_column_names):
    """Test with_seed_dataset method with basic parameters."""
    datastore_settings = DatastoreSettings(endpoint="https://huggingface.co", token="test-token")
    with patch("data_designer.config.config_builder.fetch_seed_dataset_column_names") as mock_fetch:
        mock_fetch.return_value = ["id", "name", "age", "city"]
        result = stub_empty_builder.with_seed_dataset(
            DatastoreSeedDatasetReference(dataset="test-repo/test-data.parquet", datastore_settings=datastore_settings)
        )

    assert result is stub_empty_builder
    assert stub_empty_builder.get_seed_config().dataset == "test-repo/test-data.parquet"
    assert len(stub_empty_builder.get_columns_of_type(DataDesignerColumnType.SEED_DATASET)) == 4


def test_with_seed_dataset_sampling_strategy(stub_empty_builder, mock_fetch_seed_dataset_column_names):
    """Test with_seed_dataset with different sampling strategies."""
    datastore_settings = DatastoreSettings(endpoint="https://huggingface.co", token="test-token")

    with patch("data_designer.config.config_builder.fetch_seed_dataset_column_names") as mock_fetch:
        mock_fetch.return_value = ["id", "name", "age", "city"]
        stub_empty_builder.with_seed_dataset(
            DatastoreSeedDatasetReference(dataset="test-repo/test-data.parquet", datastore_settings=datastore_settings),
            sampling_strategy=SamplingStrategy.SHUFFLE,
        )

    seed_config = stub_empty_builder.get_seed_config()
    assert seed_config.sampling_strategy == SamplingStrategy.SHUFFLE


def test_add_model_config(stub_empty_builder):
    assert len(stub_empty_builder.model_configs) == 1
    assert stub_empty_builder.model_configs[0].alias == "stub-model"

    # Test basic model config addition with inference parameters
    new_model_config = ModelConfig(
        alias="new-model",
        model="openai/gpt-4",
        inference_parameters=InferenceParameters(
            temperature=0.7,
            top_p=0.95,
            max_tokens=1024,
        ),
    )
    result = stub_empty_builder.add_model_config(new_model_config)

    assert result is stub_empty_builder
    assert len(stub_empty_builder.model_configs) == 2
    assert stub_empty_builder.model_configs[1].alias == "new-model"
    assert stub_empty_builder.model_configs[1].model == "openai/gpt-4"
    assert stub_empty_builder.model_configs[1].inference_parameters.temperature == 0.7
    assert stub_empty_builder.model_configs[1].inference_parameters.top_p == 0.95
    assert stub_empty_builder.model_configs[1].inference_parameters.max_tokens == 1024

    # Test adding model config with provider
    provider_model_config = ModelConfig(
        alias="provider-model",
        model="anthropic/claude-3",
        provider="anthropic",
        inference_parameters=InferenceParameters(temperature=0.8),
    )
    stub_empty_builder.add_model_config(provider_model_config)

    assert len(stub_empty_builder.model_configs) == 3
    assert stub_empty_builder.model_configs[2].alias == "provider-model"
    assert stub_empty_builder.model_configs[2].provider == "anthropic"

    # Test that model configs persist in built config
    stub_empty_builder.add_column(
        name="test_column",
        column_type=DataDesignerColumnType.SAMPLER,
        sampler_type=SamplerType.UUID,
    )
    config = stub_empty_builder.build()

    assert len(config.model_configs) == 3
    assert any(mc.alias == "new-model" for mc in config.model_configs)
    assert any(mc.alias == "provider-model" for mc in config.model_configs)
    assert any(mc.alias == "stub-model" for mc in config.model_configs)


def test_add_model_config_duplicate_alias(stub_empty_builder):
    duplicate_model_config = ModelConfig(
        alias="stub-model",
        model="different/model",
        inference_parameters=InferenceParameters(temperature=0.5),
    )

    with pytest.raises(
        BuilderConfigurationError,
        match="Model configuration with alias stub-model already exists. Please delete the existing model configuration or choose a different alias.",
    ):
        stub_empty_builder.add_model_config(duplicate_model_config)

    assert len(stub_empty_builder.model_configs) == 1


def test_delete_model_config(stub_empty_builder):
    model_config_1 = ModelConfig(
        alias="model-to-delete",
        model="model/delete",
        inference_parameters=InferenceParameters(temperature=0.5),
    )
    model_config_2 = ModelConfig(
        alias="model-to-keep",
        model="model/keep",
        inference_parameters=InferenceParameters(temperature=0.6),
    )
    stub_empty_builder.add_model_config(model_config_1)
    stub_empty_builder.add_model_config(model_config_2)

    assert len(stub_empty_builder.model_configs) == 3

    # Test successful deletion
    result = stub_empty_builder.delete_model_config("model-to-delete")

    assert result is stub_empty_builder
    assert len(stub_empty_builder.model_configs) == 2
    assert all(mc.alias != "model-to-delete" for mc in stub_empty_builder.model_configs)
    assert any(mc.alias == "model-to-keep" for mc in stub_empty_builder.model_configs)
    assert any(mc.alias == "stub-model" for mc in stub_empty_builder.model_configs)

    # Test deleting nonexistent model (should not raise error)
    result = stub_empty_builder.delete_model_config("nonexistent-alias")

    assert result is stub_empty_builder
    assert len(stub_empty_builder.model_configs) == 2
