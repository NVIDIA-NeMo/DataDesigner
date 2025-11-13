# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pandas as pd
import pytest

from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.utils.visualization import display_sample_record, mask_api_key
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
def config_builder_with_validation(stub_model_configs):
    """Fixture providing a DataDesignerConfigBuilder with a validation column."""
    with patch("data_designer.config.config_builder.fetch_seed_dataset_column_names") as mock_fetch:
        mock_fetch.return_value = ["code"]

        builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)

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


def test_mask_api_key():
    # Actual API keys are masked to show last 4 characters
    assert mask_api_key("sk-1234567890") == "***7890"
    assert mask_api_key("nv-some-api-key") == "***-key"

    # Short API keys (4 or fewer chars) show only asterisks
    assert mask_api_key("sk-1") == "***"
    assert mask_api_key("key") == "***"

    # Environment variable names (all uppercase) are kept visible
    assert mask_api_key("OPENAI_API_KEY") == "OPENAI_API_KEY"
    assert mask_api_key("NVIDIA_API_KEY") == "NVIDIA_API_KEY"

    # None or empty returns "(not set)"
    assert mask_api_key(None) == "(not set)"
    assert mask_api_key("") == "(not set)"
