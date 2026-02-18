# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.utils.errors import DatasetSampleDisplayError
from data_designer.config.utils.visualization import (
    display_sample_record,
    get_truncated_list_as_string,
    mask_api_key,
)
from data_designer.config.validator_params import CodeValidatorParams
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


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


def test_get_truncated_list_as_string():
    assert get_truncated_list_as_string([1, 2, 3, 4, 5]) == "[1, 2, ...]"
    assert get_truncated_list_as_string([1, 2, 3, 4, 5], max_items=1) == "[1, ...]"
    assert get_truncated_list_as_string([1, 2, 3, 4, 5], max_items=3) == "[1, 2, 3, ...]"
    assert get_truncated_list_as_string([1, 2, 3, 4, 5], max_items=10) == "[1, 2, 3, 4, 5]"
    with pytest.raises(ValueError):
        get_truncated_list_as_string([1, 2, 3, 4, 5], max_items=-1)
    with pytest.raises(ValueError):
        get_truncated_list_as_string([1, 2, 3, 4, 5], max_items=0)


def test_display_sample_record_save_html(
    validation_output: dict, config_builder_with_validation: DataDesignerConfigBuilder, tmp_path: Path
) -> None:
    """Test that display_sample_record can save output as an HTML file."""
    sample_record = {"code": "print('hello world')", "code_validation_result": validation_output}
    record_series = pd.Series(sample_record)
    save_path = tmp_path / "output.html"

    display_sample_record(record_series, config_builder_with_validation, save_path=save_path)

    assert save_path.exists()
    content = save_path.read_text()
    assert "<html" in content.lower() or "<!doctype" in content.lower()


def test_display_sample_record_save_svg(
    validation_output: dict, config_builder_with_validation: DataDesignerConfigBuilder, tmp_path: Path
) -> None:
    """Test that display_sample_record can save output as an SVG file."""
    sample_record = {"code": "print('hello world')", "code_validation_result": validation_output}
    record_series = pd.Series(sample_record)
    save_path = tmp_path / "output.svg"

    display_sample_record(record_series, config_builder_with_validation, save_path=save_path)

    assert save_path.exists()
    content = save_path.read_text()
    assert "<svg" in content.lower()


def test_display_sample_record_save_invalid_extension(
    validation_output: dict, config_builder_with_validation: DataDesignerConfigBuilder, tmp_path: Path
) -> None:
    """Test that display_sample_record raises an error for unsupported file extensions."""
    sample_record = {"code": "print('hello world')", "code_validation_result": validation_output}
    record_series = pd.Series(sample_record)
    save_path = tmp_path / "output.txt"

    with pytest.raises(DatasetSampleDisplayError, match="must be either .html or .svg"):
        display_sample_record(record_series, config_builder_with_validation, save_path=save_path)


def test_display_sample_record_save_path_none_default(
    validation_output: dict, config_builder_with_validation: DataDesignerConfigBuilder, tmp_path: Path
) -> None:
    """Test that display_sample_record with save_path=None prints to console without creating files."""
    sample_record = {"code": "print('hello world')", "code_validation_result": validation_output}
    record_series = pd.Series(sample_record)

    display_sample_record(record_series, config_builder_with_validation, save_path=None)

    # No files should be created in tmp_path
    assert list(tmp_path.iterdir()) == []
