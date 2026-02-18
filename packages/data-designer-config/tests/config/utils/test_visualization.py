# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.utils.errors import DatasetSampleDisplayError
from data_designer.config.utils.visualization import (
    WithRecordSamplerMixin,
    _apply_html_post_processing,
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
def validation_output() -> dict:
    """Fixture providing a sample validation output structure."""
    return {
        "is_valid": True,
        "python_linter_messages": [],
        "python_linter_score": 10.0,
        "python_linter_severity": "none",
    }


@pytest.fixture
def config_builder_with_validation(stub_model_configs: list) -> DataDesignerConfigBuilder:
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


def test_display_sample_record_twice_no_errors(
    validation_output: dict, config_builder_with_validation: DataDesignerConfigBuilder
) -> None:
    """Test that calling display_sample_record twice on validation output produces no errors."""
    sample_record = {"code": "print('hello world')", "code_validation_result": validation_output}
    record_series = pd.Series(sample_record)

    display_sample_record(record_series, config_builder_with_validation)
    display_sample_record(record_series, config_builder_with_validation)


def test_mask_api_key() -> None:
    """Test API key masking for various input types."""
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


def test_get_truncated_list_as_string() -> None:
    """Test list truncation for display."""
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
    """Test that display_sample_record saves HTML with dark-mode style block injected."""
    sample_record = {"code": "print('hello world')", "code_validation_result": validation_output}
    record_series = pd.Series(sample_record)
    save_path = tmp_path / "output.html"

    display_sample_record(record_series, config_builder_with_validation, save_path=save_path)

    assert save_path.exists()
    content = save_path.read_text()
    assert "<html" in content.lower() or "<!doctype" in content.lower()
    assert '<style id="data-designer-styles">' in content
    assert "color-scheme: dark" in content
    assert '<script id="data-designer-theme-listener">' in content


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

    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# _apply_html_post_processing direct tests
# ---------------------------------------------------------------------------


def test_apply_html_post_processing_injects_style(tmp_path: Path) -> None:
    """Test that _apply_html_post_processing injects a style block and theme listener before </head>."""
    html_file = tmp_path / "test.html"
    html_file.write_text("<html><head><title>Test</title></head><body></body></html>", encoding="utf-8")

    _apply_html_post_processing(html_file)

    content = html_file.read_text()
    assert '<style id="data-designer-styles">' in content
    assert "color-scheme: dark" in content
    assert '<script id="data-designer-theme-listener">' in content
    assert content.index("data-designer-styles") < content.index("</head>")


def test_apply_html_post_processing_no_head_fallback(tmp_path: Path) -> None:
    """Test that viewport tag, CSS, and theme listener are prepended when the file has no </head> tag."""
    html_file = tmp_path / "test.html"
    html_file.write_text("<body><p>Hello</p></body>", encoding="utf-8")

    _apply_html_post_processing(html_file)

    content = html_file.read_text()
    assert '<style id="data-designer-styles">' in content
    assert '<script id="data-designer-theme-listener">' in content
    assert content.startswith("<meta")


def test_apply_html_post_processing_idempotent(tmp_path: Path) -> None:
    """Test that calling _apply_html_post_processing twice does not duplicate injected content."""
    html_file = tmp_path / "test.html"
    html_file.write_text("<html><head></head><body></body></html>", encoding="utf-8")

    _apply_html_post_processing(html_file)
    _apply_html_post_processing(html_file)

    content = html_file.read_text()
    assert content.count('<style id="data-designer-styles">') == 1
    assert content.count('id="data-designer-theme-listener"') == 1


def test_apply_html_post_processing_light_theme_skips_dark_css(tmp_path: Path) -> None:
    """Test that theme='light' injects viewport tag and theme listener, but no dark style block."""
    html_file = tmp_path / "test.html"
    html_file.write_text("<html><head></head><body></body></html>", encoding="utf-8")

    _apply_html_post_processing(html_file, theme="light")

    content = html_file.read_text()
    assert '<meta name="viewport"' in content
    assert '<style id="data-designer-styles">' not in content
    assert '<script id="data-designer-theme-listener">' in content


def test_apply_html_post_processing_always_injects_viewport(tmp_path: Path) -> None:
    """Test that viewport meta tag is always injected regardless of theme."""
    for theme in ("dark", "light"):
        html_file = tmp_path / f"test_{theme}.html"
        html_file.write_text("<html><head></head><body></body></html>", encoding="utf-8")

        _apply_html_post_processing(html_file, theme=theme)

        content = html_file.read_text()
        assert '<meta name="viewport"' in content


def test_save_console_output_svg_no_dark_mode(
    validation_output: dict, config_builder_with_validation: DataDesignerConfigBuilder, tmp_path: Path
) -> None:
    """Test that SVG files do not receive dark mode CSS injection."""
    sample_record = {"code": "print('hello world')", "code_validation_result": validation_output}
    record_series = pd.Series(sample_record)
    save_path = tmp_path / "output.svg"

    display_sample_record(record_series, config_builder_with_validation, save_path=save_path)

    content = save_path.read_text()
    assert "data-designer-styles" not in content
    assert "color-scheme: dark" not in content


def test_mixin_out_of_bounds_raises_display_error(
    validation_output: dict, config_builder_with_validation: DataDesignerConfigBuilder
) -> None:
    """Test that an out-of-bounds index raises DatasetSampleDisplayError, not UnboundLocalError."""

    class FakeResults(WithRecordSamplerMixin):
        def __init__(self, dataset: pd.DataFrame, config_builder: DataDesignerConfigBuilder) -> None:
            self.dataset = dataset
            self._config_builder = config_builder
            self.dataset_metadata = None

    sample_record = {"code": "print('hello world')", "code_validation_result": validation_output}
    dataset = pd.DataFrame([sample_record])
    results = FakeResults(dataset, config_builder_with_validation)

    with pytest.raises(DatasetSampleDisplayError, match="out of bounds"):
        results.display_sample_record(index=999)
