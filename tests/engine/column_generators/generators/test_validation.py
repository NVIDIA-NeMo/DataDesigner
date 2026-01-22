# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from data_designer.config.column_configs import ValidationColumnConfig
from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.validator_params import (
    CodeValidatorParams,
    LocalCallableValidatorParams,
    RemoteValidatorParams,
    ValidatorType,
)
from data_designer.engine.column_generators.generators.validation import (
    ValidationColumnGenerator,
    get_validator_from_params,
)
from data_designer.engine.errors import DataDesignerRuntimeError
from data_designer.engine.validators import (
    LocalCallableValidator,
    PythonValidator,
    RemoteValidator,
    SQLValidator,
    ValidationResult,
)
from data_designer.engine.validators.base import ValidationOutput
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd


@pytest.mark.parametrize(
    "validator_type,validator_params,expected_class",
    [
        (ValidatorType.CODE, CodeValidatorParams(code_lang=CodeLang.PYTHON), PythonValidator),
        (ValidatorType.CODE, CodeValidatorParams(code_lang=CodeLang.SQL_SQLITE), SQLValidator),
        (ValidatorType.REMOTE, RemoteValidatorParams(endpoint_url="http://example.com/validate"), RemoteValidator),
        (
            ValidatorType.LOCAL_CALLABLE,
            LocalCallableValidatorParams(validation_function=lambda x: x),
            LocalCallableValidator,
        ),
    ],
)
def test_get_validator_from_params(validator_type, validator_params, expected_class):
    validator = get_validator_from_params(validator_type, validator_params)
    assert isinstance(validator, expected_class)


def test_get_validator_from_params_invalid_validator_type():
    with pytest.raises(AttributeError):
        get_validator_from_params("INVALID_TYPE", {})


@pytest.fixture
def stub_code_config():
    return ValidationColumnConfig(
        name="test_column",
        validator_type=ValidatorType.CODE,
        validator_params=CodeValidatorParams(code_lang=CodeLang.PYTHON),
        target_columns=["col1"],
    )


@patch("data_designer.engine.column_generators.generators.validation.get_validator_from_params", autospec=True)
def test_validation_column_generator_generate_successful_validation(
    mock_get_validator, stub_code_config, stub_resource_provider, stub_sample_dataframe
):
    mock_validator = Mock()
    mock_validator.run_validation.return_value = ValidationResult(
        data=[
            ValidationOutput(is_valid=True),
            ValidationOutput(is_valid=True),
            ValidationOutput(is_valid=True),
            ValidationOutput(is_valid=True),
        ]
    )
    mock_get_validator.return_value = mock_validator

    generator = ValidationColumnGenerator(config=stub_code_config, resource_provider=stub_resource_provider)
    result = generator.generate(stub_sample_dataframe)

    assert "test_column" in result.columns
    assert len(result["test_column"]) == 4
    mock_get_validator.assert_called_once()


@patch("data_designer.engine.column_generators.generators.validation.get_validator_from_params", autospec=True)
def test_validation_column_generator_generate_with_multiple_target_columns(mock_get_validator, stub_resource_provider):
    mock_validator = Mock()
    mock_validator.run_validation.return_value = ValidationResult(
        data=[ValidationOutput(is_valid=True), ValidationOutput(is_valid=False)]
    )
    mock_get_validator.return_value = mock_validator

    config = ValidationColumnConfig(
        name="validation_column",
        validator_type=ValidatorType.CODE,
        validator_params=CodeValidatorParams(code_lang=CodeLang.PYTHON),
        target_columns=["col1", "col2"],
    )

    generator = ValidationColumnGenerator(config=config, resource_provider=stub_resource_provider)
    df = pd.DataFrame({"col1": [1, 2], "col2": [4, 5]})

    result = generator.generate(df)

    assert "validation_column" in result.columns
    assert len(result["validation_column"]) == 2


@pytest.mark.parametrize(
    "error_message,expected_match",
    [
        (
            "Target columns.*missing_col.*are missing in dataset",
            "Target columns.*missing_col.*are missing in dataset",
        ),
        ("Validation failed", "Validation failed"),
        ("Batch validation failed", "Batch validation failed"),
    ],
)
@patch("data_designer.engine.column_generators.generators.validation.get_validator_from_params", autospec=True)
def test_validation_column_generator_generate_error_cases(
    mock_get_validator, stub_resource_provider, stub_sample_dataframe, error_message, expected_match
):
    if "missing_col" in error_message:
        config = ValidationColumnConfig(
            name="validation_column",
            validator_type=ValidatorType.CODE,
            validator_params=CodeValidatorParams(code_lang=CodeLang.PYTHON),
            target_columns=["missing_col"],
        )
        generator = ValidationColumnGenerator(config=config, resource_provider=stub_resource_provider)
    else:
        mock_validator = Mock()
        mock_validator.run_validation.side_effect = Exception(error_message)
        mock_get_validator.return_value = mock_validator

        config = ValidationColumnConfig(
            name="validation_column",
            validator_type=ValidatorType.CODE,
            validator_params=CodeValidatorParams(code_lang=CodeLang.PYTHON),
            target_columns=["col1"],
        )
        generator = ValidationColumnGenerator(config=config, resource_provider=stub_resource_provider)

    with pytest.raises(Exception, match=expected_match):
        generator.generate(stub_sample_dataframe)


@pytest.mark.parametrize(
    "validator_type,max_parallel_requests,batch_size,expected_call_count",
    [
        (ValidatorType.REMOTE, 2, 1, 2),  # Parallel execution (actual behavior)
        (ValidatorType.REMOTE, 1, 1, 2),  # Sequential execution
        (ValidatorType.LOCAL_CALLABLE, None, 1, 3),
        (ValidatorType.CODE, None, 2, 3),  # Batching with 5 records
    ],
)
@patch("data_designer.engine.column_generators.generators.validation.get_validator_from_params", autospec=True)
def test_validation_column_generator_generate_with_different_strategies(
    mock_get_validator,
    stub_resource_provider,
    validator_type,
    max_parallel_requests,
    batch_size,
    expected_call_count,
):
    mock_validator = Mock()
    mock_validator.run_validation.return_value = ValidationResult(data=[ValidationOutput(is_valid=True)])
    mock_get_validator.return_value = mock_validator

    if validator_type == ValidatorType.REMOTE:
        validator_params = RemoteValidatorParams(
            endpoint_url="http://example.com/validate",
            max_parallel_requests=max_parallel_requests,
        )
        df = pd.DataFrame({"col1": [1, 2]})
    elif validator_type == ValidatorType.LOCAL_CALLABLE:
        validator_params = LocalCallableValidatorParams(validation_function=lambda x: x)
        df = pd.DataFrame({"col1": [1, 2, 3]})
    else:
        validator_params = CodeValidatorParams(code_lang=CodeLang.PYTHON)
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})

    config = ValidationColumnConfig(
        name="validation_column",
        validator_type=validator_type,
        validator_params=validator_params,
        target_columns=["col1"],
        batch_size=batch_size,
    )

    generator = ValidationColumnGenerator(config=config, resource_provider=stub_resource_provider)

    result = generator.generate(df)
    assert mock_validator.run_validation.call_count == expected_call_count

    assert "validation_column" in result.columns
    assert len(result["validation_column"]) == len(df)


@patch("data_designer.engine.column_generators.generators.validation.get_validator_from_params", autospec=True)
def test_validation_column_generator_validate_in_parallel_failure(mock_get_validator, stub_resource_provider):
    mock_validator = Mock()
    mock_validator.run_validation.return_value = ValidationResult(data=[ValidationOutput(is_valid=True)])
    mock_get_validator.return_value = mock_validator

    config = ValidationColumnConfig(
        name="validation_column",
        validator_type=ValidatorType.REMOTE,
        validator_params=RemoteValidatorParams(
            endpoint_url="http://example.com/validate",
            max_parallel_requests=2,
        ),
        target_columns=["col1"],
        batch_size=1,
    )

    generator = ValidationColumnGenerator(config=config, resource_provider=stub_resource_provider)
    df = pd.DataFrame({"col1": [1, 2]})

    with patch(
        "data_designer.engine.column_generators.generators.validation.ConcurrentThreadExecutor"
    ) as mock_executor_class:
        mock_executor = Mock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor

        def mock_submit(func, batch, context):
            pass

        mock_executor.submit.side_effect = mock_submit

        with pytest.raises(
            DataDesignerRuntimeError,
            match="Validation task failed due to an unexpected error in parallel execution",
        ):
            generator.generate(df)

        call_kwargs = mock_executor_class.call_args[1]
        assert call_kwargs["disable_early_shutdown"] == stub_resource_provider.run_config.disable_early_shutdown
