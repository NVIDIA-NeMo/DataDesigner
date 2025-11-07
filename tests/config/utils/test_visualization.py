# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import pytest

from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.utils.visualization import display_sample_record
from data_designer.config.validator_params import CodeValidatorParams


@pytest.fixture
def validation_output():
    """Fixture providing a sample validation output structure."""
    return {
        "is_valid": True,
        "python_linter_messages": [],
        "python_linter_score": 10.0,
        "python_linter_severity": "none",
    }


@pytest.fixture
def config_builder_with_validation():
    """Fixture providing a DataDesignerConfigBuilder with a validation column."""
    builder = DataDesignerConfigBuilder()

    # Add a validation column configuration
    builder.add_column(
        name="code_validation_result",
        column_type="validation",
        target_columns=["code"],
        validator_type="code",
        validator_params=CodeValidatorParams(code_lang=CodeLang.PYTHON),
    )

    return builder


def test_display_sample_record_twice_no_errors(validation_output, config_builder_with_validation):
    """Test that calling display_sample_record twice on validation output produces no errors."""
    # Create a sample record with the validation output
    sample_record = {"code": "print('hello world')", "code_validation_result": validation_output}

    # Convert to pandas Series to match expected input format
    record_series = pd.Series(sample_record)

    # Call display_sample_record twice - should not produce any errors
    display_sample_record(record_series, config_builder_with_validation)
    display_sample_record(record_series, config_builder_with_validation)

    # If we reach this point without exceptions, the test passes
    assert True
